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
    print(f"{name:<20}  {time_s*1e6:>6.1f} us  {bandwidth:>6.1f} GB/s", file=sys.stderr)


def create_lut8_benchmark(
    name: str, bits: int, fn: Callable[[Tensor, Tensor, Tensor, Tensor], None]
) -> Callable[[int, int, int, int], None]:
    assert 8 % bits == 0, "bits must divide 8"

    def _run(k: int, n: int, copies: int, reps: int) -> None:
        torch.manual_seed(42)
        a = torch.randn((copies, k), device="cuda", dtype=torch.bfloat16)
        bq = torch.randint(
            0, 256, (copies, n, (k * bits) // 8), device="cuda", dtype=torch.uint8
        )
        lut = torch.linspace(-1, 1, 2**bits, device="cuda", dtype=torch.bfloat16)
        lut8 = torch.cartesian_prod(*[lut] * (8 // bits)).view(256, -1)
        out = torch.empty((copies, n), device="cuda", dtype=torch.bfloat16)
        run_benchmark(
            name,
            sizeof(a, bq, out) // copies,
            lambda i: fn(a[i % copies], bq[i % copies], lut8, out[i % copies]),
            reps=reps,
        )

    return _run


def sizeof(*tensors: Tensor) -> int:
    return sum(t.element_size() * t.nelement() for t in tensors)


def assert_rmsen(expected: Tensor, actual: Tensor, tol: float) -> None:
    expected_sse = expected.float().pow(2).sum()
    rmsen = (actual - expected).float().pow(2).sum().div(expected_sse).sqrt()
    assert rmsen < tol, f"rmsen={rmsen}"


# Implementations

### mv


# a   :: [k, dtype]
# b   :: [n, k, dtype]
# out :: [n, dtype]
@triton.jit
def kernel__mv(a_ptr, b_ptr, out_ptr, k, BLOCK_SIZE: tl.constexpr) -> None:
    i_n = tl.program_id(axis=0)
    acc = 0.0
    for ik0 in range(0, k, BLOCK_SIZE):
        ik = ik0 + tl.arange(0, BLOCK_SIZE)
        a = tl.load(a_ptr + ik)
        b = tl.load(b_ptr + i_n * k + ik)
        prod = a.to(tl.float32) * b.to(tl.float32)
        acc += tl.sum(prod, axis=0)
    tl.store(out_ptr + i_n, acc)


def run_mv(a: Tensor, b: Tensor, out: Tensor) -> None:
    block_size = min(2048, a.shape[0])
    n, k = b.shape
    assert a.shape == (k,)
    assert out.shape == (n,)
    assert k % block_size == 0
    grid = (n,)
    kernel__mv[grid](a, b, out, k, BLOCK_SIZE=block_size)


def test_mv() -> None:
    n, k = 128, 4096
    torch.manual_seed(42)
    a = torch.randn((k,), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((n,), device="cuda", dtype=torch.bfloat16)
    run_mv(a, b, out)
    assert_rmsen(a @ b.T, out, tol=1e-3)


def benchmark_mv(k: int, n: int, copies: int, reps: int) -> None:
    a = torch.randn((copies, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((copies, n, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((copies, n), device="cuda", dtype=torch.bfloat16)
    run_benchmark(
        "mv",
        sizeof(a, b, out) // copies,
        lambda i: run_mv(a[i % copies], b[i % copies], out[i % copies]),
        reps=reps,
    )


### mv_4b_lut8_ref


@torch.compile(mode="max-autotune-no-cudagraphs")
def mv_lut_ref(
    a: Tensor, bq: Tensor, lut8: Tensor, out: Tensor | None = None
) -> Tensor:
    return torch.matmul(a, lut8[bq.long()].flatten(start_dim=1).T, out=out)


benchmark_mv_8b_lut8_ref = create_lut8_benchmark("mv_8b_lut8_ref", 8, mv_lut_ref)
benchmark_mv_4b_lut8_ref = create_lut8_benchmark("mv_4b_lut8_ref", 4, mv_lut_ref)
benchmark_mv_2b_lut8_ref = create_lut8_benchmark("mv_2b_lut8_ref", 2, mv_lut_ref)
benchmark_mv_1b_lut8_ref = create_lut8_benchmark("mv_1b_lut8_ref", 1, mv_lut_ref)


### mv_4b_lut8


# a   :: [k, dtype]
# b   :: [n, k//2, uint8]
# lut :: [256, 2, dtype]
# out :: [n, dtype]
@triton.jit
def kernel__mv_4b_lut8(
    a_ptr,
    b_ptr,
    lut_ptr,
    out_ptr,
    k,
    ELEMENTS_PER_BYTE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    n = tl.program_id(axis=0)
    ai_ptr = a_ptr + tl.arange(0, ELEMENTS_PER_BYTE * BLOCK_SIZE)
    bi_ptr = b_ptr + n * (k // ELEMENTS_PER_BYTE) + tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    for i in range(0, k // BLOCK_SIZE // ELEMENTS_PER_BYTE):
        a = tl.load(ai_ptr)
        qb = tl.load(bi_ptr)
        b = tl.reshape(
            tl.load(
                lut_ptr
                + ELEMENTS_PER_BYTE * tl.cast(qb[:, None], tl.int16)
                + tl.arange(0, ELEMENTS_PER_BYTE)
            ),
            ELEMENTS_PER_BYTE * BLOCK_SIZE,
        )
        acc += tl.sum(a.to(tl.float32) * b.to(tl.float32))
        ai_ptr += ELEMENTS_PER_BYTE * BLOCK_SIZE
        bi_ptr += BLOCK_SIZE
    tl.store(out_ptr + n, acc)


def run_mv_4b_lut8(a: Tensor, b: Tensor, lut: Tensor, out: Tensor) -> None:
    elements_per_byte = lut.shape[1]
    block_size = min(2048, a.shape[0] // elements_per_byte)
    n, k2 = b.shape
    k = elements_per_byte * k2
    assert a.dtype == lut.dtype
    assert b.dtype == torch.uint8
    assert lut.shape == (256, elements_per_byte)
    assert a.shape == (k,)
    assert out.shape == (n,)
    assert k % (elements_per_byte * block_size) == 0
    grid = (n,)
    kernel__mv_4b_lut8[grid](
        a,
        b,
        lut.contiguous(),
        out,
        k,
        ELEMENTS_PER_BYTE=elements_per_byte,
        BLOCK_SIZE=block_size,
    )


def test_mv_4b_lut8() -> None:
    n, k = 128, 4096
    torch.manual_seed(42)
    a = torch.randn((k,), device="cuda", dtype=torch.bfloat16)
    bq = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    lut4 = torch.linspace(-1, 1, 16, device="cuda", dtype=torch.bfloat16)
    lut8 = torch.cartesian_prod(lut4, lut4)
    out = torch.empty((n,), device="cuda", dtype=torch.bfloat16)
    run_mv_4b_lut8(a, bq, lut8, out)
    assert_rmsen(mv_lut_ref(a, bq, lut8), out, tol=1e-3)


benchmark_mv_8b_lut8 = create_lut8_benchmark("mv_8b_lut8", 8, run_mv_4b_lut8)
benchmark_mv_4b_lut8 = create_lut8_benchmark("mv_4b_lut8", 4, run_mv_4b_lut8)
benchmark_mv_2b_lut8 = create_lut8_benchmark("mv_2b_lut8", 2, run_mv_4b_lut8)
benchmark_mv_1b_lut8 = create_lut8_benchmark("mv_1b_lut8", 1, run_mv_4b_lut8)


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

    top_vars = list(globals().items())

    # Run test_*
    if args.profile is None:
        for name, fn in top_vars:
            if callable(fn) and name.startswith("test_"):
                fn()

    # Run benchmark_*
    for name, fn in top_vars:
        if callable(fn) and name.startswith("benchmark_"):
            if args.profile is None or args.profile == name.replace("benchmark_", ""):
                fn(k=args.k, n=args.n, copies=args.copies, reps=args.reps)


if __name__ == "__main__":
    main()
