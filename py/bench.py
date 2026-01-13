"""Triton implementations and benchmarks."""

import sys
from dataclasses import dataclass
from functools import partial
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


def sizeof(*tensors: Tensor) -> int:
    return sum(t.element_size() * t.nelement() for t in tensors)


def assert_rmsen(expected: Tensor, actual: Tensor, tol: float) -> None:
    expected_sse = expected.float().pow(2).sum()
    rmsen = (actual - expected).float().pow(2).sum().div(expected_sse).sqrt()
    assert rmsen < tol, f"rmsen={rmsen}"


# Benchmarks


@dataclass
class Settings:
    m: int
    k: int
    n: int
    g: int
    copies: int
    reps: int


class Benchmark:
    def __init__(self, name: str):
        self.name = name
        assert name not in BENCHMARKS, f"Benchmark {name} already registered"
        BENCHMARKS[name] = self

    def bytes_rw(self, s: Settings) -> int:
        raise NotImplementedError

    def ops(self, s: Settings) -> int:
        raise NotImplementedError

    def setup(self, s: Settings) -> Callable[[], None]:
        raise NotImplementedError

    def run(self, s: Settings) -> None:
        torch.manual_seed(42)
        fn = self.setup(s)
        time_s = measure_time(fn, s.reps)
        bandwidth = self.bytes_rw(s) / time_s / 1e9
        ops_per_s = self.ops(s) / time_s / 1e12
        print(
            f"{self.name:<25}  {time_s*1e6:>6.1f} us  {bandwidth:>6.1f} GB/s  {ops_per_s:>6.1f} TFLOPS",
            file=sys.stderr,
        )


BENCHMARKS: dict[str, Benchmark] = {}


class RegisterBenchmark_mv(Benchmark):
    def __init__(
        self,
        name: str,
        fn: Callable[[Tensor, Tensor, Tensor], None],
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(name)
        self.fn = fn
        self.dtype = dtype

    def bytes_rw(self, s: Settings) -> int:
        return (s.k + s.n * s.k + s.n) * self.dtype.itemsize

    def ops(self, s: Settings) -> int:
        return 2 * s.n * s.k

    def setup(self, s: Settings) -> Callable[[], None]:
        a = torch.randn((s.copies, s.k), device="cuda", dtype=self.dtype)
        b = torch.randn((s.copies, s.n, s.k), device="cuda", dtype=self.dtype)
        out = torch.empty((s.copies, s.n), device="cuda", dtype=self.dtype)
        return lambda i: self.fn(
            a[i % s.copies], b[i % s.copies], out=out[i % s.copies]
        )


class RegisterBenchmark_mv_lut8(Benchmark):
    def __init__(
        self,
        name: str,
        fn: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], None],
        bits: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        assert 8 % bits == 0, "bits must divide 8"
        super().__init__(name)
        self.fn = fn
        self.bits = bits
        self.dtype = dtype

    def bytes_rw(self, s: Settings) -> int:
        # Don't count the LUT, as we don't use `copies` for that
        b_size = s.n * (s.k * self.bits) // 8
        a_bs_out_size = (s.k + s.n * (s.k // s.g) + s.n) * self.dtype.itemsize
        return a_bs_out_size + b_size

    def ops(self, s: Settings) -> int:
        return 2 * s.n * s.k

    def setup(self, s: Settings) -> Callable[[], None]:
        assert s.k % s.g == 0, "k must be divisible by g"
        torch.manual_seed(42)
        a = torch.randn((s.copies, s.k), device="cuda", dtype=torch.bfloat16)
        bq = torch.randint(
            0,
            256,
            (s.copies, s.n, (s.k * self.bits) // 8),
            device="cuda",
            dtype=torch.uint8,
        )
        lut = torch.linspace(-1, 1, 2**self.bits, device="cuda", dtype=torch.bfloat16)
        lut8 = torch.cartesian_prod(*[lut] * (8 // self.bits)).view(256, -1)
        bs = torch.rand(
            (s.copies, s.n, s.k // s.g), device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty((s.copies, s.n), device="cuda", dtype=torch.bfloat16)
        return lambda i: self.fn(
            a[i % s.copies], bq[i % s.copies], lut8, bs[i % s.copies], out[i % s.copies]
        )


class RegisterBenchmark_mm(Benchmark):
    def __init__(
        self,
        name: str,
        fn: Callable[[Tensor, Tensor, Tensor], None],
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(name)
        self.fn = fn
        self.dtype = dtype

    def bytes_rw(self, s: Settings) -> int:
        return (s.m * s.k + s.n * s.k + s.m * s.n) * self.dtype.itemsize

    def ops(self, s: Settings) -> int:
        return 2 * s.m * s.k * s.n

    def setup(self, s: Settings) -> Callable[[], None]:
        a = torch.randn((s.copies, s.m, s.k), device="cuda", dtype=self.dtype)
        b = torch.randn((s.copies, s.n, s.k), device="cuda", dtype=self.dtype)
        out = torch.empty((s.copies, s.m, s.n), device="cuda", dtype=self.dtype)
        return lambda i: self.fn(
            a[i % s.copies], b[i % s.copies], out=out[i % s.copies]
        )


class RegisterBenchmark_mm_lut8(Benchmark):
    def __init__(
        self,
        name: str,
        fn: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], None],
        bits: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        assert 8 % bits == 0, "bits must divide 8"
        super().__init__(name)
        self.fn = fn
        self.bits = bits
        self.dtype = dtype

    def bytes_rw(self, s: Settings) -> int:
        # Don't count the LUT, as we don't use `copies` for that
        b_size = s.n * (s.k * self.bits) // 8
        a_bs_out_size = (
            s.m * s.k + s.n * (s.k // s.g) + s.m * s.n
        ) * self.dtype.itemsize
        return a_bs_out_size + b_size

    def ops(self, s: Settings) -> int:
        return 2 * s.m * s.k * s.n

    def setup(self, s: Settings) -> Callable[[], None]:
        assert s.k % s.g == 0, "k must be divisible by g"
        torch.manual_seed(42)
        a = torch.randn((s.copies, s.m, s.k), device="cuda", dtype=torch.bfloat16)
        bq = torch.randint(
            0,
            256,
            (s.copies, s.n, (s.k * self.bits) // 8),
            device="cuda",
            dtype=torch.uint8,
        )
        lut = torch.linspace(-1, 1, 2**self.bits, device="cuda", dtype=torch.bfloat16)
        lut8 = torch.cartesian_prod(*[lut] * (8 // self.bits)).view(256, -1)
        bs = torch.rand(
            (s.copies, s.n, s.k // s.g), device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty((s.copies, s.m, s.n), device="cuda", dtype=torch.bfloat16)
        return lambda i: self.fn(
            a[i % s.copies], bq[i % s.copies], lut8, bs[i % s.copies], out[i % s.copies]
        )


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


RegisterBenchmark_mv("mv_ref", lambda a, b, out: torch.matmul(a, b.T, out=out[None]))
RegisterBenchmark_mv("mv", run_mv)


### mm_lut_ref_unscaled


@torch.compile(mode="max-autotune-no-cudagraphs")
def mm_lut_ref_unscaled(
    a: Tensor, bq: Tensor, lut8: Tensor, bs: Tensor, out: Tensor | None = None
) -> Tensor:
    b = lut8[bq.long()].flatten(start_dim=1)
    return torch.matmul(a, b.T, out=out)


RegisterBenchmark_mv_lut8("mv_8b_lut8_ref_unscaled", mm_lut_ref_unscaled, bits=8)
RegisterBenchmark_mv_lut8("mv_4b_lut8_ref_unscaled", mm_lut_ref_unscaled, bits=4)
RegisterBenchmark_mv_lut8("mv_2b_lut8_ref_unscaled", mm_lut_ref_unscaled, bits=2)
RegisterBenchmark_mv_lut8("mv_1b_lut8_ref_unscaled", mm_lut_ref_unscaled, bits=1)


### mm_lut_ref


@torch.compile(mode="max-autotune-no-cudagraphs")
def mm_lut_ref(
    a: Tensor, bq: Tensor, lut8: Tensor, bs: Tensor, out: Tensor | None = None
) -> Tensor:
    bv = lut8[bq.long()].flatten(start_dim=1)
    b = bv.view(*bs.shape, -1).mul(bs[:, :, None]).flatten(start_dim=1)
    return torch.matmul(a, b.T, out=out)


RegisterBenchmark_mv_lut8("mv_8b_lut8_ref", mm_lut_ref, bits=8)
RegisterBenchmark_mv_lut8("mv_4b_lut8_ref", mm_lut_ref, bits=4)
RegisterBenchmark_mv_lut8("mv_2b_lut8_ref", mm_lut_ref, bits=2)
RegisterBenchmark_mv_lut8("mv_1b_lut8_ref", mm_lut_ref, bits=1)


### mv_lut8


# a   :: [k, dtype]
# b   :: [n, k/E, uint8]
# lut :: [256, E, dtype]
# bs  :: [n, k/G, dtype]
# out :: [n, dtype]
@triton.jit
def kernel__mv_lut8(
    a_ptr,
    b_ptr,
    lut_ptr,
    bs_ptr,
    out_ptr,
    k,
    ELEMENTS_PER_BYTE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
) -> None:
    BLOCK_ELEMENTS: tl.constexpr = BLOCK_SIZE * ELEMENTS_PER_BYTE
    BLOCK_GROUPS: tl.constexpr = BLOCK_ELEMENTS // GROUP_SIZE
    n = tl.program_id(axis=0)

    ai_ptr = a_ptr + tl.arange(0, BLOCK_ELEMENTS)
    bi_ptr = b_ptr + n * (k // ELEMENTS_PER_BYTE) + tl.arange(0, BLOCK_SIZE)
    bsi_ptr = bs_ptr + n * (k // GROUP_SIZE) + tl.arange(0, BLOCK_GROUPS)
    out = tl.zeros((), dtype=tl.float32)
    for _ in range(0, k // BLOCK_ELEMENTS):
        a = tl.load(ai_ptr)
        bq = tl.load(bi_ptr)
        bu = tl.load(
            lut_ptr
            + ELEMENTS_PER_BYTE * tl.cast(bq[:, None], tl.int32)
            + tl.arange(0, ELEMENTS_PER_BYTE)
        )
        bs = tl.load(bsi_ptr)
        b = tl.reshape(
            tl.reshape(bu, [BLOCK_GROUPS, GROUP_SIZE]) * bs[:, None], BLOCK_ELEMENTS
        )
        out += tl.sum(a.to(tl.float32) * b.to(tl.float32))
        ai_ptr += BLOCK_ELEMENTS
        bi_ptr += BLOCK_SIZE
        bsi_ptr += BLOCK_GROUPS

    tl.store(out_ptr + n, out)


def run_mv_lut8(a: Tensor, b: Tensor, lut: Tensor, bs: Tensor, out: Tensor) -> None:
    # shape inference
    (k,) = a.shape
    (n,) = out.shape
    elements_per_byte = lut.shape[1]
    group_size = k // bs.shape[1]
    # assertions
    assert a.dtype == lut.dtype == bs.dtype == out.dtype
    assert b.dtype == torch.uint8
    assert b.shape == (n, k // elements_per_byte)
    assert bs.shape == (n, k // group_size)
    assert lut.shape == (256, elements_per_byte)
    assert all([t.is_contiguous() for t in [a, b, lut, bs, out]])
    # heuristic
    block_size = min(2048, k // elements_per_byte)
    assert k % (elements_per_byte * block_size) == 0
    assert block_size % group_size == 0
    # kernel
    grid = (n,)
    kernel__mv_lut8[grid](
        a,
        b,
        lut,
        bs,
        out,
        k,
        ELEMENTS_PER_BYTE=elements_per_byte,
        GROUP_SIZE=group_size,
        BLOCK_SIZE=block_size,
    )


def test_mv_4b_lut8() -> None:
    # testing 4 bits per element
    n, k, g = 128, 4096, 64
    torch.manual_seed(42)
    a = torch.randn((k,), device="cuda", dtype=torch.bfloat16)
    bq = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    lut4 = torch.linspace(-1, 1, 16, device="cuda", dtype=torch.bfloat16)
    lut8 = torch.cartesian_prod(lut4, lut4)
    bs = torch.rand((n, k // g), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((n,), device="cuda", dtype=torch.bfloat16)
    run_mv_lut8(a, bq, lut8, bs, out)
    assert_rmsen(mm_lut_ref(a, bq, lut8, bs), out, tol=1e-2)


RegisterBenchmark_mv_lut8("mv_8b_lut8", run_mv_lut8, bits=8)
RegisterBenchmark_mv_lut8("mv_4b_lut8", run_mv_lut8, bits=4)
RegisterBenchmark_mv_lut8("mv_2b_lut8", run_mv_lut8, bits=2)
RegisterBenchmark_mv_lut8("mv_1b_lut8", run_mv_lut8, bits=1)

### mm


def _autotune_configs() -> list[triton.Config]:
    def _cfg(bm: int, bn: int, bk: int, gm: int, s: int, w: int) -> triton.Config:
        return triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": gm,
            },
            num_stages=s,
            num_warps=w,
        )

    return [
        # BLK_M, BLK_N, BLK_K, GRP_M, s=STAGES, w=WARPS
        _cfg(128, 256, 64, 8, s=3, w=8),
        _cfg(64, 256, 32, 8, s=4, w=4),
        _cfg(128, 128, 32, 8, s=4, w=4),
        _cfg(128, 64, 32, 8, s=4, w=4),
        _cfg(64, 128, 32, 8, s=4, w=4),
        _cfg(128, 32, 32, 8, s=4, w=4),
        _cfg(64, 32, 32, 8, s=5, w=2),
        _cfg(32, 64, 32, 8, s=5, w=2),
        # Good config for fp8 inputs.
        _cfg(128, 256, 128, 8, s=3, w=8),
        _cfg(256, 128, 128, 8, s=3, w=8),
        _cfg(256, 64, 128, 8, s=4, w=4),
        _cfg(64, 256, 128, 8, s=4, w=4),
        _cfg(128, 128, 128, 8, s=4, w=4),
        _cfg(128, 64, 64, 8, s=4, w=4),
        _cfg(64, 128, 64, 8, s=4, w=4),
        _cfg(128, 32, 64, 8, s=4, w=4),
    ]


# a :: [m, k, dtype]
# b :: [n, k, dtype]
# out :: [m, n, dtype]
@triton.autotune(configs=_autotune_configs(), key=["m", "n", "k", "DTYPE"])
@triton.jit
def kernel__mm(
    a_ptr,
    b_ptr,
    out_ptr,
    # Matrix dimensions
    m,
    n,
    k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
) -> None:
    # PID mapping (with grouped ordering for L2 cache reuse)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Integer bound assumptions to guide backend optimizations
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(m > 0)
    tl.assume(n > 0)
    tl.assume(k > 0)

    # Block pointers for traversing a, b and writing out
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_N, BLOCK_SIZE_K] pointers
    # `out_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_N] pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % n
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_am[:, None] * k + offs_k[None, :])
    b_ptrs = b_ptr + (offs_bn[:, None] * k + offs_k[None, :])
    out_ptrs = out_ptr + offs_outm[:, None] * n + offs_outn[None, :]

    # Compute a block of `out`, iterating over the k dimension.
    out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ik in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k - ik * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < k - ik * BLOCK_SIZE_K, other=0.0)
        out = tl.dot(a, tl.trans(b), out)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    # Write out the block of `out`
    tl.store(out_ptrs, out, mask=(offs_outm[:, None] < m) & (offs_outn[None, :] < n))


def run_mm(a: Tensor, b: Tensor, out: Tensor) -> None:
    m, k = a.shape
    n, _ = b.shape
    assert b.shape == (n, k)
    assert out.shape == (m, n)
    assert a.dtype == b.dtype == out.dtype
    assert all([t.is_contiguous() for t in [a, b, out]])
    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )
    kernel__mm[grid](a, b, out, m, n, k)


def test_mm() -> None:
    m, k, n = 192, 4096, 128
    torch.manual_seed(42)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    run_mm(a, b, out)
    assert_rmsen(a @ b.T, out, tol=1e-3)


RegisterBenchmark_mm("mm_ref", lambda a, b, out: torch.matmul(a, b.T, out=out))
RegisterBenchmark_mm("mm", run_mm)


### mm_lut_ref_unscaled, mm_lut_ref (mm benchmarks)

RegisterBenchmark_mm_lut8("mm_8b_lut8_ref_unscaled", mm_lut_ref_unscaled, bits=8)
RegisterBenchmark_mm_lut8("mm_4b_lut8_ref_unscaled", mm_lut_ref_unscaled, bits=4)
RegisterBenchmark_mm_lut8("mm_2b_lut8_ref_unscaled", mm_lut_ref_unscaled, bits=2)
RegisterBenchmark_mm_lut8("mm_1b_lut8_ref_unscaled", mm_lut_ref_unscaled, bits=1)

RegisterBenchmark_mm_lut8("mm_8b_lut8_ref", mm_lut_ref, bits=8)
RegisterBenchmark_mm_lut8("mm_4b_lut8_ref", mm_lut_ref, bits=4)
RegisterBenchmark_mm_lut8("mm_2b_lut8_ref", mm_lut_ref, bits=2)
RegisterBenchmark_mm_lut8("mm_1b_lut8_ref", mm_lut_ref, bits=1)


### mm_lut8


# a :: [m, k, dtype]
# b :: [n, k//E, uint8]
# lut :: [256, E, dtype]
# out :: [m, n, dtype]
@triton.autotune(
    configs=[
        # Hand-tuned for 4b, (m, n, k) = (16, 4096, 4096) on L40S
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 8,
            },
            num_stages=1,
            num_warps=4,
        ),
        # Hand-tuned for 4b, (m, n, k) = (256, 4096, 4096) on L40S
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=1,
            num_warps=1,
        ),
    ],
    key=["m", "n", "k", "ELEMENTS_PER_BYTE", "DTYPE"],
)
@triton.jit
def kernel__mm_lut(
    a_ptr,
    b_ptr,
    lut_ptr,
    bs_ptr,
    out_ptr,
    # Matrix dimensions
    m,
    n,
    k,
    ELEMENTS_PER_BYTE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
) -> None:
    BLOCK_K_ELEMENTS: tl.constexpr = BLOCK_SIZE_K * ELEMENTS_PER_BYTE
    tl.static_assert(BLOCK_K_ELEMENTS % GROUP_SIZE == 0)
    BLOCK_K_GROUPS: tl.constexpr = BLOCK_K_ELEMENTS // GROUP_SIZE

    # PID mapping (with grouped ordering for L2 cache reuse)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Integer bound assumptions to guide backend optimizations
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(m > 0)
    tl.assume(n > 0)
    tl.assume(k > 0)

    # Block pointers for traversing a, b and writing out
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_N, BLOCK_SIZE_K] pointers
    # `out_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_N] pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % n
    offs_ak = tl.arange(0, BLOCK_K_ELEMENTS)
    offs_bk = tl.arange(0, BLOCK_SIZE_K)
    offs_bsk = tl.arange(0, BLOCK_K_GROUPS)
    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_am[:, None] * k + offs_ak[None, :])
    b_ptrs = b_ptr + (offs_bn[:, None] * (k // ELEMENTS_PER_BYTE) + offs_bk[None, :])
    bs_ptrs = bs_ptr + (offs_bn[:, None] * (k // GROUP_SIZE) + offs_bsk[None, :])
    out_ptrs = out_ptr + offs_outm[:, None] * n + offs_outn[None, :]

    # Compute a block of `out`, iterating over the k dimension.
    out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ik in range(0, tl.cdiv(k, BLOCK_K_ELEMENTS)):
        # Attempt 0 - hangs the compiler
        a = tl.reshape(  # [BLOCK_SIZE_M, BLOCK_K_GROUPS, GROUP_SIZE]
            tl.load(
                a_ptrs,
                mask=offs_ak[None, :] < k - ik * BLOCK_K_ELEMENTS,
                other=0.0,
            ),
            [BLOCK_SIZE_M, BLOCK_K_GROUPS, GROUP_SIZE],
        )
        # Note: should be safe to load without mask since lut[any_byte] is defined,
        # and multiplying by `a` will mask when k is out of bounds.
        qb = tl.load(b_ptrs)
        bu = tl.reshape(  # [BLOCK_SIZE_N, BLOCK_K_GROUPS, GROUP_SIZE]
            tl.load(
                lut_ptr
                + ELEMENTS_PER_BYTE * tl.cast(qb[:, :, None], tl.int32)
                + tl.arange(0, ELEMENTS_PER_BYTE),
            ),
            [BLOCK_SIZE_N, BLOCK_K_GROUPS, GROUP_SIZE],
        )
        bs = tl.load(bs_ptrs)  # [BLOCK_SIZE_N, BLOCK_K_GROUPS]
        ab = tl.dot(  # [BLOCK_K_GROUPS, BLOCK_SIZE_M, BLOCK_SIZE_N]
            tl.trans(a, 1, 0, 2), tl.trans(bu, 1, 2, 0)
        )
        out = out + tl.sum(ab * tl.trans(bs)[:, None, :], axis=0)

        a_ptrs += BLOCK_K_ELEMENTS
        b_ptrs += BLOCK_SIZE_K
        bs_ptrs += BLOCK_K_GROUPS

    # Write out the block of `out`
    tl.store(out_ptrs, out, mask=(offs_outm[:, None] < m) & (offs_outn[None, :] < n))


def run_mm_lut(a: Tensor, b: Tensor, lut: Tensor, bs: Tensor, out: Tensor) -> None:
    # shape inference
    m, k = a.shape
    _, n = out.shape
    elements_per_byte = lut.shape[1]
    group_size = k // bs.shape[1]
    # assertions
    assert a.dtype == lut.dtype == bs.dtype == out.dtype
    assert b.dtype == torch.uint8
    assert b.shape == (n, k // elements_per_byte)
    assert bs.shape == (n, k // group_size)
    assert lut.shape == (256, elements_per_byte)
    assert out.shape == (m, n)
    assert all([t.is_contiguous() for t in [a, b, lut, out]])
    # kernel
    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )
    kernel__mm_lut[grid](
        a,
        b,
        lut,
        bs,
        out,
        m,
        n,
        k,
        ELEMENTS_PER_BYTE=elements_per_byte,
        GROUP_SIZE=group_size,
    )


def test_mm_4b_lut8() -> None:
    torch.manual_seed(42)
    m, k, n, g = 192, 4096, 768, 64
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    lut4 = torch.linspace(-1, 1, 16, device="cuda", dtype=torch.bfloat16)
    lut8 = torch.cartesian_prod(lut4, lut4)
    bs = torch.rand((n, k // g), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    run_mm_lut(a, b, lut8, bs, out=out)
    assert_rmsen(mm_lut_ref(a, b, lut8, bs), out, tol=1e-2)  # tol too high?


RegisterBenchmark_mm_lut8("mm_8b_lut8", run_mm_lut, bits=8)
RegisterBenchmark_mm_lut8("mm_4b_lut8", run_mm_lut, bits=4)
RegisterBenchmark_mm_lut8("mm_2b_lut8", run_mm_lut, bits=2)
RegisterBenchmark_mm_lut8("mm_1b_lut8", run_mm_lut, bits=1)


# Top-level


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profile", nargs="?", help="Select a specific method to profile"
    )
    parser.add_argument("-m", default=16, type=int, help="Dimension m (`mm` only)")
    parser.add_argument("-k", default=4096, type=int, help="Dimension k")
    parser.add_argument("-n", default=4096, type=int, help="Dimension n")
    parser.add_argument("-g", default=64, type=int, help="Dimension g (group size)")
    parser.add_argument("--copies", default=100, type=int, help="Number of arg copies")
    parser.add_argument("--reps", default=1000, type=int, help="Number of reps")
    args = parser.parse_args()

    # Run test_*
    top_vars = list(globals().items())
    if args.profile is None:
        for name, fn in top_vars:
            if callable(fn) and name.startswith("test_"):
                fn()

    # Run benchmarks
    settings = Settings(
        m=args.m, k=args.k, n=args.n, g=args.g, copies=args.copies, reps=args.reps
    )
    for name, benchmark in BENCHMARKS.items():
        if args.profile is None or args.profile == name:
            benchmark.run(settings)


if __name__ == "__main__":
    main()
