"""Model-level quantisation and benchmarking."""

import functools
import gc
import itertools
import re
import sys
from dataclasses import dataclass

import torch
import transformers
from torch import Tensor, nn

import qbench
import qkernels

# Quantised models


@dataclass
class QuantisationConfig:
    bits: int
    block_size: int
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
        weight_grouped = weight.unflatten(-1, (-1, c.block_size))
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
        multipliers = 2 ** torch.arange(
            0, 8, c.bits, device=device, dtype=torch.uint8
        ).flip(0)
        weight_data = (
            weight_indices.to(torch.uint8)
            .unflatten(-1, (-1, elements_per_byte))
            .mul(multipliers)
            .sum(-1, dtype=torch.uint8)
        )
        lut = torch.cartesian_prod(*([lut1] * elements_per_byte)).view(256, -1)
        return cls(weight.shape[1], weight.shape[0], weight_data, scale, lut)


def test_quantised_linear() -> None:
    torch.manual_seed(100)
    w = torch.randn((192, 256), device="cuda", dtype=torch.bfloat16)
    x = torch.randn(100, w.shape[1], device="cuda", dtype=torch.bfloat16)
    for bits, tol in [(8, 1e-2), (4, 2e-1), (2, 0.75), (1, 1)]:
        q = QuantisedLinear.create(w, QuantisationConfig(bits=bits, block_size=64))
        y = q(x)
        qkernels._assert_rmsen(x @ w.T, y, tol=tol)


def quantise(
    module: nn.Module,
    config: QuantisationConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """In-place quantisation of all nn.Linear layers in a module.

    Note: any parameter sharing is lost.
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

            weight = child.weight.data.to(device=device, dtype=dtype)
            setattr(module, name, QuantisedLinear.create(weight, config))

        else:  # recurse
            quantise(child, config, device=device, dtype=dtype)

    module.to(device=device, dtype=dtype)  # convert non-quantised modules
    return module


# Benchmarking helpers


class Generator:
    """A graph-captureable autoregressive text generator."""

    def __init__(
        self, model: transformers.PreTrainedModel, prompt_ids: Tensor, steps: int
    ) -> None:
        batch_size, prompt_n = prompt_ids.shape
        max_length = prompt_n + steps + 1  # +1 for prefill, which happens before step()
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
    avg_time: float
    avg_time_stderr: float
    output: list[str]

    def __str__(self) -> str:
        bandwidth = self.bytes_rw / self.avg_time / 1e9
        ops_per_s = self.ops / self.avg_time / 1e12
        return "  ".join(
            [
                f"{self.batch_size / self.avg_time:>6.1f} tokens/s",
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


def count_parameter_read_bytes(model: transformers.PreTrainedModel) -> int:
    """Approximate bytes read per token."""
    bytes_r = 0
    for module in model.modules():
        # Ignore sparse modules (embeddings), with negligible reads
        if not isinstance(module, nn.Embedding):
            bytes_r += sum(
                p.numel() * p.element_size() for p in module.parameters(recurse=False)
            )
    return bytes_r


def benchmark(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    batch_size: int,
    reps: int,
    prompt: str = "The",
) -> Result:
    """Measure generation performance of a model."""
    with torch.no_grad():
        prompt_ids = (
            tokenizer([prompt] * batch_size, return_tensors="pt")
            .to(model.device)
            .input_ids
        )
        # +3 as measure_time runs 3 "warmup/capture" iterations
        generator = Generator(model, prompt_ids, reps + 3)
        avg_time, avg_time_stderr = qbench.measure_time(
            lambda _: generator.step(), inner_reps=1, outer_reps=reps
        )
        # approximate operations and bytes transferred
        return Result(
            batch_size=batch_size,
            bytes_rw=count_parameter_read_bytes(model),
            ops=batch_size * count_ops(model),
            avg_time=avg_time,
            avg_time_stderr=avg_time_stderr,
            output=tokenizer.batch_decode(generator.tokens),
        )


# Custom model configs

custom_llama_configs = {
    "4B": dict(
        dim=3072,
        n_layers=24,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_mask_type="block_causal",
        attn_type="flex",
    ),
    "11B": dict(
        dim=4096,
        n_layers=48,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_mask_type="block_causal",
        attn_type="flex",
    ),
    "30B": dict(
        dim=6144,
        n_layers=60,
        n_heads=48,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_mask_type="block_causal",
        attn_type="flex",
    ),
}


def get_custom_llama_config(key: str) -> transformers.PretrainedConfig:
    config = custom_llama_configs[key].copy()

    # defaults
    vocab_size = 128256  # Llama 3
    tie_word_embeddings = False

    # computed
    dim = config.pop("dim")
    ffn_dim_multiplier = config.pop("ffn_dim_multiplier")
    multiple_of = config.pop("multiple_of")
    intermediate_size = multiple_of * (
        (int(4 * 2 / 3 * dim * ffn_dim_multiplier) + multiple_of - 1) // multiple_of
    )

    # unneeded for inference
    del config["attn_mask_type"]
    del config["attn_type"]

    transformers_config = transformers.models.llama.modeling_llama.LlamaConfig(
        _name_or_path=f"custom-llama-{key}",
        vocab_size=vocab_size,
        hidden_size=dim,
        num_hidden_layers=config.pop("n_layers"),
        num_attention_heads=config.pop("n_heads"),
        num_key_value_heads=config.pop("n_kv_heads"),
        intermediate_size=intermediate_size,
        rope_theta=config.pop("rope_theta"),
        dtype=torch.bfloat16,
        tie_word_embeddings=tie_word_embeddings,
    )
    assert not config, f"unused keys {list(config)}"
    return transformers_config


# Top-level


@dataclass
class BenchmarkSettings:
    model: str
    batch_size: int
    quantisation: QuantisationConfig | None
    reps: int

    def __str__(self) -> str:
        q = (
            f"b={self.quantisation.block_size} {self.quantisation.bits}-bit"
            if self.quantisation is not None
            else "16-bit"
        )
        return f"{self.model} batch={self.batch_size} {q}"


def _estimate_fits_on_device(
    settings: BenchmarkSettings, device: torch.device, dtype: torch.dtype
) -> bool:
    _, total_mem_bytes = torch.cuda.mem_get_info(device.index)
    if m := re.search(r"(\d[0-9.]*)[bB]$", settings.model):
        model_params_bn = float(m[1])
        bytes_per_param = (
            dtype.itemsize
            if settings.quantisation is None
            else settings.quantisation.bits / 8
            + dtype.itemsize / settings.quantisation.block_size
        )
        estimated_model_bytes = model_params_bn * 1e9 * bytes_per_param
        return estimated_model_bytes < total_mem_bytes * 0.8  # Leave some headroom
    print(f"No size for model {settings.model} - assuming it fits", file=sys.stderr)
    return True


@functools.lru_cache()
def get_custom_llama_tokenizer() -> transformers.PreTrainedTokenizerBase:
    return transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")


def run_benchmark(settings: BenchmarkSettings, log: qbench.Log) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    device, dtype = torch.device("cuda"), torch.bfloat16

    out = dict(**settings.__dict__)
    if out["quantisation"] is not None:  # flatten
        out.update(out["quantisation"].__dict__)
    del out["quantisation"]

    try:
        if not _estimate_fits_on_device(settings, device, dtype):
            raise RuntimeError(
                "skipped; model size * bits/param is too large for the device"
            )

        if settings.model.startswith("custom-llama-"):
            key = settings.model.replace("custom-llama-", "")
            # Create in float32, since bfloat16 RNG on CPU is slow
            model = transformers.AutoModelForCausalLM.from_config(
                get_custom_llama_config(key), dtype=torch.float32
            )
            tokenizer = get_custom_llama_tokenizer()
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                settings.model, dtype=dtype
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(model.name_or_path)

        out.update(parameters=sum(p.numel() for p in model.parameters()))
        if settings.quantisation is not None:
            quantise(model, settings.quantisation, device=device, dtype=dtype)

        model.to(device=device, dtype=dtype)
        result = benchmark(model, tokenizer, settings.batch_size, settings.reps)
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
        default=["custom-llama-4B"],
        nargs="+",
        help=f"Model ids, e.g. 'custom-llama-4B' (sizes: {list(custom_llama_configs)})"
        " or 'Qwen/Qwen3-4B'",
    )
    parser.add_argument(
        "--batch-size",
        default=[1],
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
        "--block-size",
        default=[64],
        type=int,
        nargs="+",
        help="Quantisation block sizes",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=64,
        help="Number of generation steps to measure",
    )
    args = parser.parse_args()
    transformers.utils.logging.disable_progress_bar()

    test_quantised_linear()

    with qbench.Log("models") as log:
        keys = ["model", "batch_size", "bits", "block_size"]
        for values in itertools.product(*(getattr(args, k) for k in keys)):
            s = dict(zip(keys, values))
            quantisation = (
                None
                if s["bits"] == 16
                else QuantisationConfig(bits=s["bits"], block_size=s["block_size"])
            )
            settings = BenchmarkSettings(
                model=s["model"],
                batch_size=s["batch_size"],
                quantisation=quantisation,
                reps=args.reps,
            )
            run_benchmark(settings, log)


if __name__ == "__main__":
    _main()
