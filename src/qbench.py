"""Triton implementations and benchmarks."""

import datetime
import gc
import itertools
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import triton
from torch import Tensor

import qkernels
import marlin

# Utilities


LOG_DIR = Path("out/dev")


def _current_timestamp() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")


def _meta() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    return dict(
        device=torch.cuda.get_device_name(torch.cuda.current_device()),
        cuda_version=torch.version.cuda,
        torch_version=torch.__version__,
        triton_version=triton.__version__,
        commit=commit,
    )


class Log:
    def __init__(self, key: str) -> None:
        self.id = f"{key}-{_current_timestamp()}"
        self.meta = dict(id=self.id, **_meta())
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / f"{self.id}.jsonl"
        self.file = self.path.open("w", encoding="utf-8")

    def __call__(self, entry: dict[str, Any]) -> None:
        print(json.dumps({**entry, **self.meta}), file=self.file, flush=True)

    def __enter__(self) -> "Log":
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> None:
        self.file.close()


def measure_time(
    fn: Callable[[int], None], inner_reps: int, outer_reps: int
) -> tuple[float, float]:
    fn(0)  # Initial warmup (autotune & JIT)

    # Further warmup & capture a graph
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(torch.cuda.Stream()):
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for i in range(inner_reps):
                fn(i)
    torch.cuda.synchronize()

    # One final warmup replay
    g.replay()
    torch.cuda.synchronize()

    # Measure time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(outer_reps):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms / inner_reps / 1000)

    atimes = torch.tensor(times)
    return atimes.mean().item(), atimes.std().item() / math.sqrt(len(atimes))


def sizeof(*tensors: Tensor) -> int:
    return sum(t.element_size() * t.nelement() for t in tensors)


# Benchmarks


@dataclass
class Settings:
    m: int
    k: int
    n: int
    g: int
    bits: int
    copies: int
    inner_reps: int
    outer_reps: int

    def __post_init__(self) -> None:
        assert (
            self.copies <= self.inner_reps
        ), "insufficient inner_reps to cycle through all copies"

    def __str__(self) -> str:
        return f"m={self.m} k={self.k} n={self.n} g={self.g} bits={self.bits}"


@dataclass
class Result:
    name: str
    bytes_rw: int
    ops: int
    time_s: float
    stderr_s: float

    def __str__(self) -> str:
        bandwidth = self.bytes_rw / self.time_s / 1e9
        ops_per_s = self.ops / self.time_s / 1e12
        return "  ".join(
            [
                f"{self.name:<25}",
                f"{self.time_s*1e6:>6.1f} us",
                f"{bandwidth:>6.1f} GB/s",
                f"{ops_per_s:>6.1f} TFLOPS",
            ]
        )


class UnsupportedSettings(Exception):
    pass


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

    def run(self, s: Settings) -> Result:
        torch.manual_seed(42)
        fn = self.setup(s)
        time_s, stderr_s = measure_time(fn, s.inner_reps, s.outer_reps)
        return Result(
            name=self.name,
            bytes_rw=self.bytes_rw(s),
            ops=self.ops(s),
            time_s=time_s,
            stderr_s=stderr_s,
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
        if s.m != 1 or s.bits != 16:
            raise UnsupportedSettings("RegisterBenchmark_mv only supports m=1, bits=16")
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
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(name)
        self.fn = fn
        self.dtype = dtype

    def bytes_rw(self, s: Settings) -> int:
        # Don't count the LUT, as we don't use `copies` for that
        b_size = s.n * (s.k * s.bits) // 8
        a_bs_out_size = (s.k + s.n * (s.k // s.g) + s.n) * self.dtype.itemsize
        return a_bs_out_size + b_size

    def ops(self, s: Settings) -> int:
        return 2 * s.n * s.k

    def setup(self, s: Settings) -> Callable[[], None]:
        if s.m != 1 or s.bits not in [1, 2, 4, 8]:
            raise UnsupportedSettings(
                "RegisterBenchmark_mv only supports m=1, bits in [1, 2, 4, 8]"
            )
        assert s.k % s.g == 0, "k must be divisible by g"
        torch.manual_seed(42)
        a = torch.randn((s.copies, s.k), device="cuda", dtype=torch.bfloat16)
        bq = torch.randint(
            0,
            256,
            (s.copies, s.n, (s.k * s.bits) // 8),
            device="cuda",
            dtype=torch.uint8,
        )
        lut = torch.linspace(-1, 1, 2**s.bits, device="cuda", dtype=torch.bfloat16)
        lut8 = torch.cartesian_prod(*[lut] * (8 // s.bits)).view(256, -1)
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
        if s.bits != 16:
            raise UnsupportedSettings("RegisterBenchmark_mm only supports bits=16")
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
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(name)
        self.fn = fn
        self.dtype = dtype

    def bytes_rw(self, s: Settings) -> int:
        # Don't count the LUT, as we don't use `copies` for that
        b_size = s.n * (s.k * s.bits) // 8
        a_bs_out_size = (
            s.m * s.k + s.n * (s.k // s.g) + s.m * s.n
        ) * self.dtype.itemsize
        return a_bs_out_size + b_size

    def ops(self, s: Settings) -> int:
        return 2 * s.m * s.k * s.n

    def setup(self, s: Settings) -> Callable[[], None]:
        if s.bits not in [1, 2, 4, 8]:
            raise UnsupportedSettings(
                "RegisterBenchmark_mm only supports bits in [1, 2, 4, 8]"
            )
        assert s.k % s.g == 0, "k must be divisible by g"
        torch.manual_seed(42)
        a = torch.randn((s.copies, s.m, s.k), device="cuda", dtype=torch.bfloat16)
        bq = torch.randint(
            0,
            256,
            (s.copies, s.n, (s.k * s.bits) // 8),
            device="cuda",
            dtype=torch.uint8,
        )
        lut = torch.linspace(-1, 1, 2**s.bits, device="cuda", dtype=torch.bfloat16)
        lut8 = torch.cartesian_prod(*[lut] * (8 // s.bits)).view(256, -1)
        bs = torch.rand(
            (s.copies, s.n, s.k // s.g), device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty((s.copies, s.m, s.n), device="cuda", dtype=torch.bfloat16)
        return lambda i: self.fn(
            a[i % s.copies], bq[i % s.copies], lut8, bs[i % s.copies], out[i % s.copies]
        )


class RegisterBenchmark_marlin(Benchmark):
    def __init__(self):
        super().__init__("marlin")

    def bytes_rw(self, s: Settings) -> int:
        # Don't count the LUT or workspace, as we don't use `copies` for that
        b_size = s.n * (s.k * 4) // 8
        a_bs_out_size = (
            s.m * s.k + s.n * (s.k // s.g) + s.m * s.n
        ) * torch.float16.itemsize
        return a_bs_out_size + b_size

    def ops(self, s: Settings) -> int:
        return 2 * s.m * s.k * s.n

    def setup(self, s: Settings) -> Callable[[], None]:
        if s.bits != 4 or s.g not in [64, 128]:
            raise UnsupportedSettings(
                "RegisterBenchmark_marlin only supports bits=4, g=64 or 128"
            )
        assert s.k % s.g == 0, "k must be divisible by g"
        torch.manual_seed(42)
        a = torch.randn((s.copies, s.m, s.k), device="cuda", dtype=torch.float16)
        ii = torch.iinfo(torch.int32)
        b = torch.randint(
            ii.min,
            ii.max,
            (s.copies, s.k // 16, 2 * s.n),
            device="cuda",
            dtype=torch.int32,
        )
        bs = torch.randn(
            (s.copies, s.k // s.g, s.n), device="cuda", dtype=torch.float16
        )
        out = torch.empty((s.copies, s.m, s.n), device="cuda", dtype=torch.float16)
        centroids = torch.randn((256, 2), device="cuda", dtype=torch.float16)
        workspace = torch.zeros(s.n // 128 * 16, dtype=torch.int)
        return lambda i: marlin.mul(
            a[i % s.copies],
            b[i % s.copies],
            out[i % s.copies],
            bs[i % s.copies],
            centroids,
            workspace,
        )


# Additional implementations


@torch.compile(mode="max-autotune-no-cudagraphs")
def mm_lut_ref_unscaled(
    a: Tensor, bq: Tensor, lut8: Tensor, bs: Tensor, out: Tensor | None = None
) -> Tensor:
    b = lut8[bq.long()].flatten(start_dim=1)
    return torch.matmul(a, b.T, out=out)


# Registrations

# mv
RegisterBenchmark_mv("mv_ref", lambda a, b, out: torch.matmul(a, b.T, out=out[None]))
RegisterBenchmark_mv("mv", qkernels.run_mv)
RegisterBenchmark_mv_lut8("mv_lut8_ref_unscaled", mm_lut_ref_unscaled)
RegisterBenchmark_mv_lut8("mv_lut8_ref", qkernels.mm_lut_ref)
RegisterBenchmark_mv_lut8("mv_lut8", qkernels.run_mv_lut)

# mm
RegisterBenchmark_mm("mm_ref", lambda a, b, out: torch.matmul(a, b.T, out=out))
RegisterBenchmark_mm("mm", qkernels.run_mm)
RegisterBenchmark_mm_lut8("mm_lut8_ref_unscaled", mm_lut_ref_unscaled)
RegisterBenchmark_mm_lut8("mm_lut8_ref", qkernels.mm_lut_ref)
RegisterBenchmark_mm_lut8("mm_lut8", qkernels.run_mm_lut)

# marlin
RegisterBenchmark_marlin()


# Top-level


def run_tests() -> None:
    tests = [
        (k, v)
        for k, v in itertools.chain(qkernels.__dict__.items(), globals().items())
        if k.startswith("test_")
    ]
    for name, fn in tests:
        if callable(fn) and name.startswith("test_"):
            fn()


def run_benchmarks(
    settings: Settings, only: str, include: str, exclude: str, log: Log
) -> None:

    gc.collect()
    torch.cuda.empty_cache()
    header_printed = False
    for name, benchmark in BENCHMARKS.items():
        if only != "" and only != name:
            continue
        if include != "" and not re.search(include, name):
            continue
        if exclude != "" and re.search(exclude, name):
            continue
        try:
            result = benchmark.run(settings)
            if not header_printed:
                print(f"# {settings}", file=sys.stderr)
                header_printed = True
            print(result, file=sys.stderr)
            log(
                dict(
                    test=name,
                    **settings.__dict__,
                    bytes_rw=result.bytes_rw,
                    ops=result.ops,
                    avg_time=result.time_s,
                    avg_time_stderr=result.stderr_s,
                )
            )
        except UnsupportedSettings:
            pass
    if header_printed:
        print(file=sys.stderr)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", default=[1, 4, 16, 64, 256], type=int, nargs="+", help="Dimension m"
    )
    parser.add_argument("-k", default=[4096], type=int, nargs="+", help="Dimension k")
    parser.add_argument(
        "-n", default=["k"], nargs="+", help="Dimension n (int or 'k' for square)"
    )
    parser.add_argument(
        "-g", default=[64], type=int, nargs="+", help="Dimension g (block size)"
    )
    parser.add_argument(
        "-b", "--bits", default=[16, 4, 1], type=int, nargs="+", help="Bits per element"
    )
    parser.add_argument("--copies", default=100, type=int, help="Number of arg copies")
    parser.add_argument("--inner-reps", default=100, type=int, help="Number of reps")
    parser.add_argument(
        "--outer-reps", default=100, type=int, help="Number of rep samples"
    )
    # Selection
    parser.add_argument(
        "--profile", default="", help="Select a specific method to profile (skip tests)"
    )
    parser.add_argument("--include", default="", help="Methods to include (regex)")
    parser.add_argument("--exclude", default="_ref", help="Methods to exclude (regex)")
    args = parser.parse_args()

    if args.profile == "":
        run_tests()

    with Log("py") as log:
        keys = ["m", "k", "n", "g", "bits"]
        for values in itertools.product(*(getattr(args, k) for k in keys)):
            s = dict(zip(keys, values))
            if s["n"] == "k":
                s["n"] = s["k"]
            run_benchmarks(
                Settings(
                    **s,
                    copies=args.copies,
                    inner_reps=args.inner_reps,
                    outer_reps=args.outer_reps,
                ),
                only=args.profile,
                include=args.include,
                exclude=args.exclude,
                log=log,
            )


if __name__ == "__main__":
    _main()
