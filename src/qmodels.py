"""Model-level quantisation and benchmarking."""

from dataclasses import dataclass
import gc
import itertools
import sys

import torch
import transformers
from torch import Tensor, nn

import qbench
import qkernels

# Quantised models


@dataclass
class QuantisationConfig:
    bits: int
    group_size: int
    kernel: str = "triton"


class QuantisedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_data: Tensor,
        scale: Tensor,
        lut: Tensor,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_data = nn.Parameter(weight_data, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.lut = nn.Buffer(lut)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim > 2:
            return self.forward(x.flatten(end_dim=-2)).unflatten(0, x.shape[:-1])
        out = torch.empty(
            (x.shape[0], self.weight_data.shape[0]), device=x.device, dtype=x.dtype
        )
        if x.shape[0] == 1:
            qkernels.run_mv_lut(x[0], self.weight_data, self.lut, self.scale, out[0])
        else:
            qkernels.run_mm_lut(x, self.weight_data, self.lut, self.scale, out)
        return out

    @classmethod
    def create(cls, weight: Tensor, c: QuantisationConfig) -> "QuantisedLinear":
        device = weight.device
        elements_per_byte = 8 // c.bits

        # Group scaling
        weight_grouped = weight.unflatten(-1, (-1, c.group_size))
        scale = (
            weight_grouped.abs().mean(dim=-1)
            if c.bits <= 2
            else weight_grouped.abs().amax(dim=-1)
        )
        scale.clamp_min_(1e-12)
        weight_scaled = (weight_grouped / scale[..., None]).flatten(-2)

        # Quantisation
        lut1 = torch.linspace(
            -1, 1, steps=2**c.bits, device=device, dtype=torch.bfloat16
        )
        weight_indices = torch.bucketize(
            weight_scaled, (lut1[1:] + lut1[:-1]) / 2
        ).clamp_(0, 2**c.bits - 1)
        weight_data = (
            weight_indices.to(torch.uint8)
            .unflatten(-1, (-1, elements_per_byte))
            .mul(
                2
                ** torch.arange(0, 8, c.bits, device=device, dtype=torch.uint8).flip(0)
            )
            .sum(-1, dtype=torch.uint8)
        )
        lut = torch.cartesian_prod(*([lut1] * elements_per_byte)).view(256, -1)
        return cls(weight.shape[1], weight.shape[0], weight_data, scale, lut)


def test_quantised_linear() -> None:
    torch.manual_seed(100)
    w = torch.randn((192, 256), device="cuda", dtype=torch.bfloat16)
    x = torch.randn(100, w.shape[1], device="cuda", dtype=torch.bfloat16)
    for bits, tol in [(8, 1e-2), (4, 2e-1), (2, 0.75), (1, 1)]:
        q = QuantisedLinear.create(w, QuantisationConfig(bits=bits, group_size=64))
        y = q(x)
        qkernels._assert_rmsen(x @ w.T, y, tol=tol)


def quantise(
    module: nn.Module, config: QuantisationConfig, device: torch.device
) -> nn.Module:
    """In-place quantisation of all nn.Linear layers in a module.

    Any parameter sharing is not preserved.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            assert (
                type(child) == nn.Linear
            ), f"nn.Linear subclasses are not supported: {type(child)}"
            assert child.bias is None, "Bias is unsupported"
            assert (
                child.in_features % 64 == 0
            ), f"in_features must be multiple of 64, got {child.in_features}"
            setattr(
                module,
                name,
                QuantisedLinear.create(child.weight.data.to(device), config),
            )
        else:
            quantise(child, config, device)
    module.to(device)  # convert non-quantised modules
    return module


# Benchmarking helpers


class Generator:
    """A graph-captureable autoregressive text generator."""

    def __init__(
        self, model: transformers.PreTrainedModel, prompt_ids: Tensor, max_length: int
    ) -> None:
        batch_size, prompt_n = prompt_ids.shape
        self.model = model
        self.input_ids = torch.zeros(
            (batch_size, max_length), dtype=torch.long, device=model.device
        )
        self.input_ids[:, :prompt_n] = prompt_ids
        self.cache = transformers.cache_utils.StaticCache(
            model.config, max_cache_len=max_length, device=model.device
        )
        logits = model(prompt_ids, past_key_values=self.cache, use_cache=True).logits
        self.last_token = self.input_ids[:, prompt_n] = logits[:, -1].argmax(dim=-1)
        self.last_index = torch.tensor(prompt_n, device=model.device)

    def step(self) -> None:
        logits = self.model(
            input_ids=self.last_token[:, None],
            cache_position=self.last_index[None],
            past_key_values=self.cache,
            use_cache=True,
        ).logits
        self.last_token[...] = logits[:, -1].argmax(dim=-1)
        self.last_index += 1
        token = self.last_token[:, None]
        self.input_ids.scatter_(1, self.last_index[None, None].expand_as(token), token)

    @property
    def tokens(self) -> Tensor:
        return self.input_ids[:, : self.last_index + 1]


@dataclass
class Result:
    batch_size: int
    bytes_rw: int
    ops: int
    time_s: float
    stderr_s: float
    output: list[str]

    def __str__(self) -> str:
        bandwidth = self.bytes_rw / self.time_s / 1e9
        ops_per_s = self.ops / self.time_s / 1e12
        return "  ".join(
            [
                f"{self.batch_size / self.time_s:>6.1f} tokens/s",
                f"{bandwidth:>6.1f} GB/s",
                f"{ops_per_s:>6.1f} TFLOPS",
            ]
        )


def count_ops(model: transformers.PreTrainedModel) -> int:
    """Approximate ops per token, respecting quantised layers (ignoring attention)."""
    ops = 0
    for module in model.modules():
        if isinstance(module, QuantisedLinear):
            ops += 2 * module.in_features * module.out_features
        else:
            ops += sum(2 * p.numel() for p in module.parameters(recurse=False))
    return ops


def benchmark(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    batch_size: int,
    steps: int,
    prompt: str = "The",
) -> Result:
    """Measure generation performance of a model."""
    prompt_ids = (
        tokenizer([prompt] * batch_size, return_tensors="pt").to(model.device).input_ids
    )
    generator = Generator(model, prompt_ids, steps + 4)
    time_s, time_s_stderr = qbench.measure_time(
        lambda _: generator.step(), inner_reps=1, outer_reps=steps
    )
    # approximate operations and bytes transferred
    ops = batch_size * count_ops(model)
    bytes_r = sum(p.numel() * p.element_size() for p in model.parameters())
    return Result(
        batch_size=batch_size,
        bytes_rw=bytes_r,
        ops=ops,
        time_s=time_s,
        stderr_s=time_s_stderr,
        output=tokenizer.batch_decode(generator.tokens),
    )


# Top-level


@dataclass
class BenchmarkSettings:
    model: str
    batch_size: int
    quantisation: QuantisationConfig | None
    steps: int

    def __str__(self) -> str:
        q = (
            f"g={self.quantisation.group_size} {self.quantisation.bits}-bit"
            if self.quantisation is not None
            else "16-bit"
        )
        return f"{self.model} batch={self.batch_size} {q}"


def run_benchmark(settings: BenchmarkSettings, log: qbench.Log) -> Result:
    gc.collect()
    torch.cuda.empty_cache()

    out = dict(**settings.__dict__)
    if out["quantisation"] is not None:  # flatten
        out.update(out["quantisation"].__dict__)
    del out["quantisation"]

    try:
        device = torch.device("cuda")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            settings.model, dtype=torch.bfloat16
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model.name_or_path)
        out.update(parameters=sum(p.numel() for p in model.parameters()))
        with torch.no_grad():
            if settings.quantisation is not None:
                quantise(model, settings.quantisation, device)
            result = benchmark(
                model.to(device), tokenizer, settings.batch_size, settings.steps
            )
            out.update(result.__dict__)
            print(f"{str(settings):>40}:  {result}", file=sys.stderr)
    except Exception as e:
        out.update(error=repr(e))
        print(f"{str(settings):>40}:  ERROR {e}", file=sys.stderr)
    log(out)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=["Qwen/Qwen3-4B", "Qwen/Qwen3-14B"],
        nargs="+",
        help="Model ids",
    )
    parser.add_argument(
        "--batch-size",
        default=[1, 16, 64],
        type=int,
        nargs="+",
        help="Generation batch sizes",
    )
    parser.add_argument(
        "--bits",
        default=[16, 4, 1],
        type=int,
        nargs="+",
        help="Quantisation bit widths",
    )
    parser.add_argument(
        "--group-size",
        default=[64],
        type=int,
        nargs="+",
        help="Quantisation group sizes",
    )
    parser.add_argument(
        "--steps", type=int, default=64, help="Number of generation steps"
    )
    args = parser.parse_args()
    transformers.utils.logging.disable_progress_bar()

    test_quantised_linear()

    with qbench.Log("models") as log:
        keys = ["model", "batch_size", "bits", "group_size"]
        for values in itertools.product(*(getattr(args, k) for k in keys)):
            s = dict(zip(keys, values))
            quantisation = (
                None
                if s["bits"] == 16
                else QuantisationConfig(bits=s["bits"], group_size=s["group_size"])
            )
            settings = BenchmarkSettings(
                model=s["model"],
                batch_size=s["batch_size"],
                quantisation=quantisation,
                steps=args.steps,
            )
            run_benchmark(settings, log)


if __name__ == "__main__":
    _main()
