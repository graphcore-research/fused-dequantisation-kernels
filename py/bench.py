"""Triton implementations and benchmarks."""

import sys
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


def _run_benchmark(
    name: str, bytes_rw: int, ops: int, fn: Callable[[None], None], reps: int = 100
) -> None:
    time_s = measure_time(fn, reps)
    bandwidth = bytes_rw / time_s / 1e9
    ops_per_s = ops / time_s / 1e12
    print(
        f"{name:<20}  {time_s*1e6:>6.1f} us  {bandwidth:>6.1f} GB/s  {ops_per_s:>6.1f} TFLOPS",
        file=sys.stderr,
    )


def _benchmark_mv(
    mkn: tuple[int, int, int],
    copies: int,
    reps: int,
    name: str,
    fn: Callable[[Tensor, Tensor, Tensor], None],
) -> None:
    _, k, n = mkn
    torch.manual_seed(42)
    a = torch.randn((copies, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((copies, n, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((copies, n), device="cuda", dtype=torch.bfloat16)
    _run_benchmark(
        name,
        bytes_rw=sizeof(a, b, out) // copies,
        ops=2 * n * k,
        fn=lambda i: fn(a[i % copies], b[i % copies], out=out[i % copies]),
        reps=reps,
    )


def _benchmark_mv_lut8(
    mkn: tuple[int, int, int],
    copies: int,
    reps: int,
    name: str,
    bits: int,
    fn: Callable[[Tensor, Tensor, Tensor, Tensor], None],
) -> None:
    _, k, n = mkn
    assert 8 % bits == 0, "bits must divide 8"
    torch.manual_seed(42)
    a = torch.randn((copies, k), device="cuda", dtype=torch.bfloat16)
    bq = torch.randint(
        0, 256, (copies, n, (k * bits) // 8), device="cuda", dtype=torch.uint8
    )
    lut = torch.linspace(-1, 1, 2**bits, device="cuda", dtype=torch.bfloat16)
    lut8 = torch.cartesian_prod(*[lut] * (8 // bits)).view(256, -1)
    out = torch.empty((copies, n), device="cuda", dtype=torch.bfloat16)
    _run_benchmark(
        name,
        bytes_rw=sizeof(a, bq, out) // copies + sizeof(lut8),
        ops=2 * n * k,
        fn=lambda i: fn(a[i % copies], bq[i % copies], lut8, out[i % copies]),
        reps=reps,
    )


def _benchmark_mm(
    mkn: tuple[int, int, int],
    copies: int,
    reps: int,
    name: str,
    fn: Callable[[Tensor, Tensor, Tensor], None],
) -> None:
    m, k, n = mkn
    torch.manual_seed(42)
    a = torch.randn((copies, m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((copies, n, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((copies, m, n), device="cuda", dtype=torch.bfloat16)
    _run_benchmark(
        name,
        bytes_rw=sizeof(a, b, out) // copies,
        ops=2 * m * k * n,
        fn=lambda i: fn(a[i % copies], b[i % copies], out=out[i % copies]),
        reps=reps,
    )


def _benchmark_mm_lut8(
    mkn: tuple[int, int, int],
    copies: int,
    reps: int,
    name: str,
    bits: int,
    fn: Callable[[Tensor, Tensor, Tensor, Tensor], None],
) -> None:
    m, k, n = mkn
    assert 8 % bits == 0, "bits must divide 8"
    torch.manual_seed(42)
    a = torch.randn((copies, m, k), device="cuda", dtype=torch.bfloat16)
    bq = torch.randint(
        0, 256, (copies, n, (k * bits) // 8), device="cuda", dtype=torch.uint8
    )
    lut = torch.linspace(-1, 1, 2**bits, device="cuda", dtype=torch.bfloat16)
    lut8 = torch.cartesian_prod(*[lut] * (8 // bits)).view(256, -1)
    out = torch.empty((copies, m, n), device="cuda", dtype=torch.bfloat16)
    _run_benchmark(
        name,
        bytes_rw=sizeof(a, bq, out) // copies + sizeof(lut8),
        ops=2 * m * k * n,
        fn=lambda i: fn(a[i % copies], bq[i % copies], lut8, out[i % copies]),
        reps=reps,
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


benchmark_mv_ref = partial(
    _benchmark_mv,
    name="mv_ref",
    fn=lambda a, b, out: torch.matmul(a, b.T, out=out[None]),
)
benchmark_mv = partial(_benchmark_mv, name="mv", fn=run_mv)


### mm_lut_ref


@torch.compile(mode="max-autotune-no-cudagraphs")
def mm_lut_ref(
    a: Tensor, bq: Tensor, lut8: Tensor, out: Tensor | None = None
) -> Tensor:
    return torch.matmul(a, lut8[bq.long()].flatten(start_dim=1).T, out=out)


benchmark_mv_8b_lut8_ref = partial(
    _benchmark_mv_lut8, name="mv_8b_lut8_ref", bits=8, fn=mm_lut_ref
)
benchmark_mv_4b_lut8_ref = partial(
    _benchmark_mv_lut8, name="mv_4b_lut8_ref", bits=4, fn=mm_lut_ref
)
benchmark_mv_2b_lut8_ref = partial(
    _benchmark_mv_lut8, name="mv_2b_lut8_ref", bits=2, fn=mm_lut_ref
)
benchmark_mv_1b_lut8_ref = partial(
    _benchmark_mv_lut8, name="mv_1b_lut8_ref", bits=1, fn=mm_lut_ref
)


### mv_lut8


# a   :: [k, dtype]
# b   :: [n, k/E, uint8]
# lut :: [256, E, dtype]
# out :: [n, dtype]
@triton.jit
def kernel__mv_lut8(
    a_ptr,
    b_ptr,
    lut_ptr,
    out_ptr,
    k,
    ELEMENTS_PER_BYTE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    n = tl.program_id(axis=0)
    ai_ptr = a_ptr + tl.arange(0, BLOCK_SIZE * ELEMENTS_PER_BYTE)
    bi_ptr = b_ptr + n * (k // ELEMENTS_PER_BYTE) + tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    for _ in range(0, k // BLOCK_SIZE // ELEMENTS_PER_BYTE):
        a = tl.load(ai_ptr)
        qb = tl.load(bi_ptr)
        b = tl.reshape(
            tl.load(
                lut_ptr
                + ELEMENTS_PER_BYTE * tl.cast(qb[:, None], tl.int32)
                + tl.arange(0, ELEMENTS_PER_BYTE)
            ),
            BLOCK_SIZE * ELEMENTS_PER_BYTE,
        )
        acc += tl.sum(a.to(tl.float32) * b.to(tl.float32))
        ai_ptr += BLOCK_SIZE * ELEMENTS_PER_BYTE
        bi_ptr += BLOCK_SIZE
    tl.store(out_ptr + n, acc)


def run_mv_lut8(a: Tensor, b: Tensor, lut: Tensor, out: Tensor) -> None:
    elements_per_byte = lut.shape[1]
    (k,) = a.shape
    (n,) = out.shape
    block_size = min(2048, k // elements_per_byte)
    assert a.dtype == lut.dtype == out.dtype
    assert b.dtype == torch.uint8
    assert b.shape == (n, k // elements_per_byte)
    assert lut.shape == (256, elements_per_byte)
    assert k % (elements_per_byte * block_size) == 0
    grid = (n,)
    kernel__mv_lut8[grid](
        a,
        b,
        lut.contiguous(),
        out,
        k,
        ELEMENTS_PER_BYTE=elements_per_byte,
        BLOCK_SIZE=block_size,
    )


def test_mv_4b_lut8() -> None:
    # testing 4 bits per element
    n, k = 128, 4096
    torch.manual_seed(42)
    a = torch.randn((k,), device="cuda", dtype=torch.bfloat16)
    bq = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    lut4 = torch.linspace(-1, 1, 16, device="cuda", dtype=torch.bfloat16)
    lut8 = torch.cartesian_prod(lut4, lut4)
    out = torch.empty((n,), device="cuda", dtype=torch.bfloat16)
    run_mv_lut8(a, bq, lut8, out)
    assert_rmsen(mm_lut_ref(a, bq, lut8), out, tol=1e-3)


benchmark_mv_8b_lut8 = partial(
    _benchmark_mv_lut8, name="mv_8b_lut8", bits=8, fn=run_mv_lut8
)
benchmark_mv_4b_lut8 = partial(
    _benchmark_mv_lut8, name="mv_4b_lut8", bits=4, fn=run_mv_lut8
)
benchmark_mv_2b_lut8 = partial(
    _benchmark_mv_lut8, name="mv_2b_lut8", bits=2, fn=run_mv_lut8
)
benchmark_mv_1b_lut8 = partial(
    _benchmark_mv_lut8, name="mv_1b_lut8", bits=1, fn=run_mv_lut8
)

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
    # Strides
    stride_am,
    stride_bn,
    stride_outm,
    stride_outn,
    # Meta-parameters
    DTYPE: tl.constexpr,
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
    tl.assume(stride_am > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_outm > 0)
    tl.assume(stride_outn > 0)

    # Block pointers for traversing a, b and writing out
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_N, BLOCK_SIZE_K] pointers
    # `out_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_N] pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % n
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :])
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :])
    out_ptrs = (
        out_ptr + stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :]
    )

    # Compute a block of `out`, iterating over the k dimension.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ik in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k - ik * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < k - ik * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, tl.trans(b), accumulator)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    out = accumulator.to(DTYPE)

    # Write out the block of `out`
    tl.store(out_ptrs, out, mask=(offs_outm[:, None] < m) & (offs_outn[None, :] < n))


def run_mm(a: Tensor, b: Tensor, out: Tensor) -> None:
    m, k = a.shape
    n, _ = b.shape
    assert b.shape == (n, k)
    assert out.shape == (m, n)
    assert a.dtype == b.dtype == out.dtype
    assert a.is_contiguous()
    assert b.is_contiguous()
    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )
    kernel__mm[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        b.stride(0),
        out.stride(0),
        out.stride(1),
        DTYPE=getattr(tl, str(a.dtype).split(".")[-1]),
    )


def test_mm() -> None:
    m, k, n = 192, 4096, 128
    torch.manual_seed(42)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    run_mm(a, b, out)
    assert_rmsen(a @ b.T, out, tol=1e-3)


benchmark_mm_ref = partial(
    _benchmark_mm, name="mm_ref", fn=lambda a, b, out: torch.matmul(a, b.T, out=out)
)
benchmark_mm = partial(_benchmark_mm, name="mm", fn=run_mm)


### mm_lut_ref (mm benchmarks)


benchmark_mm_8b_lut8_ref = partial(
    _benchmark_mm_lut8, name="mm_8b_lut8_ref", bits=8, fn=mm_lut_ref
)
benchmark_mm_4b_lut8_ref = partial(
    _benchmark_mm_lut8, name="mm_4b_lut8_ref", bits=4, fn=mm_lut_ref
)
benchmark_mm_2b_lut8_ref = partial(
    _benchmark_mm_lut8, name="mm_2b_lut8_ref", bits=2, fn=mm_lut_ref
)
benchmark_mm_1b_lut8_ref = partial(
    _benchmark_mm_lut8, name="mm_1b_lut8_ref", bits=1, fn=mm_lut_ref
)


### mv_lut8


# a :: [m, k, dtype]
# b :: [n, k//E, uint8]
# lut :: [256, E, dtype]
# out :: [m, n, dtype]
@triton.autotune(
    configs=_autotune_configs(), key=["m", "n", "k", "ELEMENTS_PER_BYTE" "DTYPE"]
)
@triton.jit
def kernel__mm_lut(
    a_ptr,
    b_ptr,
    lut_ptr,
    out_ptr,
    # Matrix dimensions
    m,
    n,
    k,
    # Strides
    stride_am,
    stride_bn,
    stride_outm,
    stride_outn,
    # Meta-parameters
    ELEMENTS_PER_BYTE: tl.constexpr,
    DTYPE: tl.constexpr,
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
    tl.assume(stride_am > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_outm > 0)
    tl.assume(stride_outn > 0)

    # Block pointers for traversing a, b and writing out
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_N, BLOCK_SIZE_K] pointers
    # `out_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_N] pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % n
    offs_ak = tl.arange(0, BLOCK_SIZE_K * ELEMENTS_PER_BYTE)
    offs_bk = tl.arange(0, BLOCK_SIZE_K)
    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :])
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_bk[None, :])
    out_ptrs = (
        out_ptr + stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :]
    )

    # Compute a block of `out`, iterating over the k dimension.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ik in range(0, tl.cdiv(k, BLOCK_SIZE_K * ELEMENTS_PER_BYTE)):
        a = tl.load(
            a_ptrs,
            mask=offs_ak[None, :] < k - ik * BLOCK_SIZE_K * ELEMENTS_PER_BYTE,
            other=0.0,
        )
        # Note: should be safe to load without mask since lut[any_byte] is defined,
        # and multiplying by `a` will mask when k is out of bounds.
        qb = tl.load(b_ptrs)
        b = tl.reshape(
            tl.load(
                lut_ptr
                + ELEMENTS_PER_BYTE * tl.cast(qb[:, :, None], tl.int32)
                + tl.arange(0, ELEMENTS_PER_BYTE)
            ),
            [BLOCK_SIZE_N, BLOCK_SIZE_K * ELEMENTS_PER_BYTE],
        )
        accumulator = tl.dot(a, tl.trans(b), accumulator)
        a_ptrs += BLOCK_SIZE_K * ELEMENTS_PER_BYTE
        b_ptrs += BLOCK_SIZE_K
    out = accumulator.to(DTYPE)

    # Write out the block of `out`
    tl.store(out_ptrs, out, mask=(offs_outm[:, None] < m) & (offs_outn[None, :] < n))


def run_mm_lut(a: Tensor, b: Tensor, lut: Tensor, out: Tensor) -> None:
    elements_per_byte = lut.shape[1]
    m, k = a.shape
    _, n = out.shape
    assert a.dtype == lut.dtype == out.dtype
    assert b.dtype == torch.uint8
    assert b.shape == (n, k // elements_per_byte)
    assert lut.shape == (256, elements_per_byte)
    assert out.shape == (m, n)
    assert a.is_contiguous()
    assert b.is_contiguous()
    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )
    kernel__mm_lut[grid](
        a,
        b,
        lut.contiguous(),
        out,
        m,
        n,
        k,
        a.stride(0),
        b.stride(0),
        out.stride(0),
        out.stride(1),
        ELEMENTS_PER_BYTE=elements_per_byte,
        DTYPE=getattr(tl, str(a.dtype).split(".")[-1]),
    )


def test_mm_4b_lut8() -> None:
    torch.manual_seed(42)
    m, k, n = 192, 4096, 768
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    lut4 = torch.linspace(-1, 1, 16, device="cuda", dtype=torch.bfloat16)
    lut8 = torch.cartesian_prod(lut4, lut4)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    run_mm_lut(a, b, lut8, out=out)
    assert_rmsen(mm_lut_ref(a, b, lut8), out, tol=1e-2)  # tol too high?


benchmark_mm_8b_lut8 = partial(
    _benchmark_mm_lut8, name="mm_8b_lut8", bits=8, fn=run_mm_lut
)
benchmark_mm_4b_lut8 = partial(
    _benchmark_mm_lut8, name="mm_4b_lut8", bits=4, fn=run_mm_lut
)
benchmark_mm_2b_lut8 = partial(
    _benchmark_mm_lut8, name="mm_2b_lut8", bits=2, fn=run_mm_lut
)
# benchmark_mm_1b_lut8 = partial(
#     _benchmark_mm_lut8, name="mm_1b_lut8", bits=1, fn=run_mm_lut
# )

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
                fn(mkn=(args.m, args.k, args.n), copies=args.copies, reps=args.reps)


if __name__ == "__main__":
    main()
