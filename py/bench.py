"""Triton implementations and benchmarks."""

import sys
from typing import Callable

import torch
import triton
import triton.language as tl
from torch import Tensor

# Utilities


def measure_time(fn: Callable[[None], None], reps: int) -> float:
    fn(0)  # Initial warmup (autotune & JIT)

    # Further warmup & capture a graph
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    static_stream = torch.cuda.Stream()
    with torch.cuda.stream(static_stream):
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for i in range(reps):
                fn(i)
    torch.cuda.synchronize()

    # Measure time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms / reps / 1000


def run_benchmark(
    name: str, bytes_rw: int, fn: Callable[[None], None], reps: int = 100
) -> None:
    time_s = measure_time(fn, reps)
    bandwidth = bytes_rw / time_s / 1e9
    print(f"{name:<10}  {time_s*1e6:>6.1f} us  {bandwidth:>6.1f} GB/s")


def sizeof(*tensors: Tensor) -> int:
    return sum(t.element_size() * t.nelement() for t in tensors)


# Implementations


# a   :: [k, bfloat16]
# b   :: [n, k//2, uint8]
# lut :: [256, 2, bfloat16]
# out :: [n, bfloat16]
@triton.jit
def kernel__mv_4b_lut8(a_ptr, b_ptr, lut_ptr, out_ptr, k, BLOCK_SIZE: tl.constexpr):
    n = tl.program_id(axis=0)
    k2 = k // 2
    tl.assume(k2 % BLOCK_SIZE == 0)
    acc = 0.0
    for i_k2 in range(0, k2, BLOCK_SIZE):
        a = tl.load(a_ptr + 2 * i_k2 + tl.arange(0, 2 * BLOCK_SIZE))
        qb = tl.load(b_ptr + n * k2 + i_k2 + tl.arange(0, BLOCK_SIZE))
        b = tl.reshape(
            tl.load(lut_ptr + 2 * tl.cast(qb[:, None], tl.int16) + tl.arange(0, 2)),
            2 * BLOCK_SIZE,
        )
        prod = a.to(tl.float32) * b.to(tl.float32)
        acc += tl.sum(prod, axis=0)
    tl.store(out_ptr + n, acc)


def run_mv_4b_lut8(a: Tensor, b: Tensor, lut: Tensor, out: Tensor) -> None:
    block_size = min(2048, a.shape[0] // 2)
    n, k2 = b.shape
    k = 2 * k2
    assert a.dtype == lut.dtype
    assert b.dtype == torch.uint8
    assert lut.dtype.itemsize == 2
    assert lut.shape == (256, 2)
    assert a.shape == (k,)
    assert out.shape == (n,)
    assert k % (2 * block_size) == 0
    grid = (n,)
    kernel__mv_4b_lut8[grid](a, b, lut.contiguous(), out, k, BLOCK_SIZE=block_size)


def test_mv_4b_lut8() -> None:
    n, k = 128, 4096
    torch.manual_seed(42)
    a = torch.randn((k,), device="cuda", dtype=torch.bfloat16)
    bq = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    lut4 = torch.linspace(-1, 1, 16, device="cuda", dtype=torch.bfloat16)
    lut8 = torch.cartesian_prod(lut4, lut4)
    out = torch.empty((n,), device="cuda", dtype=torch.bfloat16)
    run_mv_4b_lut8(a, bq, lut8, out)
    expected = a @ lut8[bq.long()].flatten(start_dim=1).T
    rmsen = (
        (out - expected).float().pow(2).sum() / expected.float().pow(2).sum()
    ).sqrt()
    assert rmsen < 1e-3, f"rmsen={rmsen}"


def benchmark_mv_4b_lut8(k: int, n: int, copies: int, reps: int) -> None:
    a = torch.randn((copies, k), device="cuda", dtype=torch.bfloat16)
    bq = torch.randint(0, 256, (copies, n, k // 2), device="cuda", dtype=torch.uint8)
    lut4 = torch.linspace(-1, 1, 16, device="cuda", dtype=torch.bfloat16)
    lut8 = torch.cartesian_prod(lut4, lut4)
    out = torch.empty((copies, n), device="cuda", dtype=torch.bfloat16)
    run_benchmark(
        "mv_4b_lut8",
        sizeof(a, bq, out) // copies,
        lambda i: run_mv_4b_lut8(a[i % copies], bq[i % copies], lut8, out[i % copies]),
        reps=reps,
    )


# Top-level


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profile", nargs="?", help="Select a specific method to profile"
    )
    parser.add_argument("-k", default=4096, type=int, help="Dimension k")
    parser.add_argument("-n", default=4096, type=int, help="Dimension n")
    parser.add_argument("--copies", default=100, type=int, help="Number of arg copies")
    parser.add_argument("--reps", default=1000, type=int, help="Number of reps")
    args = parser.parse_args()

    # Run test_*
    if args.profile is None:
        for name, fn in globals().items():
            if callable(fn) and name.startswith("test_"):
                fn()

    # Run benchmark_*
    for name, fn in globals().items():
        if callable(fn) and name.startswith("benchmark_"):
            if args.profile is None or args.profile == name.replace("benchmark_", ""):
                fn(k=args.k, n=args.n, copies=args.copies, reps=args.reps)


if __name__ == "__main__":
    main()
