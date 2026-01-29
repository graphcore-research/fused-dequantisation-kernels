"""Model-level quantisation and benchmarking."""

import copy
import functools
import gc
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import transformers
from torch import Tensor, nn

import marlin
import qbench
import qkernels

# Quantised models


@dataclass
class QuantisationConfig:
    bits: int
    block_size: int
    kernel: Literal["triton", "torch.compile", "marlin", "marlin-lut"] = "triton"
    skip: str | None = "^lm_head$"


class QuantisedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_data: Tensor,
        scale: Tensor,
        lut: Tensor,
        kernel: Literal["triton", "torch.compile"],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_data = nn.Parameter(weight_data, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.lut = nn.Buffer(lut)
        self.kernel = kernel

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim > 2:
            return self.forward(x.flatten(end_dim=-2)).unflatten(0, x.shape[:-1])
        if self.kernel == "torch.compile":
            return qkernels.mm_lut_ref(x, self.weight_data, self.lut, self.scale)
        assert self.kernel == "triton"
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
        return cls(*weight.T.shape, weight_data, scale, lut, kernel=c.kernel)


def _create_marlin_layer(src: nn.Linear, c: QuantisationConfig) -> nn.Module:
    assert c.bits == 4, "Marlin backend only supports 4-bit quantisation"
    assert c.block_size in (64, 128), "Marlin backend requires block_size 64 or 128"
    assert src.weight.dtype == torch.half, "Marlin backend requires float16 weights"
    layer = marlin.Layer(
        infeatures=src.in_features,
        outfeatures=src.out_features,
        groupsize=c.block_size,
        is_linear={"marlin": True, "marlin-lut": False}[c.kernel],
    ).cuda()
    if not layer.is_linear:
        centroids = torch.arange(-8, 8, dtype=torch.half, device="cuda")
        layer.centroids[...] = torch.cartesian_prod(centroids, centroids).flip(1)
    weight = src.weight.data
    weight_grouped = weight.unflatten(-1, (-1, c.block_size))
    scale = weight_grouped.abs().amax(dim=-1) / 7
    scale.clamp_min_(1e-12)
    layer.pack(src, scale)
    return layer


def quantise(
    source: nn.Module,
    config: QuantisationConfig | None,
    device: torch.device,
    dtype: torch.dtype,
    key: tuple[str, ...] = (),
) -> nn.Module:
    """Quantise all nn.Linear layers in a module according to config,
    returning a new quantised module. The original module is left untouched.

    Note: any parameter sharing is lost.
    """
    should_quantise = (
        isinstance(source, nn.Linear)
        and config is not None
        and (config.skip is None or not re.search(config.skip, ".".join(key)))
    )
    if should_quantise:
        assert (
            type(source) == nn.Linear
        ), f"nn.Linear subclasses are not supported: {type(source)}"
        assert source.bias is None, "Bias is unsupported"
        assert (
            source.in_features % 64 == 0
        ), f"in_features must be multiple of 64, got {source.in_features}"

        if config.kernel.startswith("marlin"):
            module = _create_marlin_layer(copy.deepcopy(source).to(dtype), config)
        else:
            module = QuantisedLinear.create(
                source.weight.data.to(device, dtype), config
            )
    else:
        # Shallow copy the module object (not its tensors)
        module = copy.copy(source)
        module._modules = {}
        module._parameters = {}
        module._buffers = {}
        # Copy parameters and buffers directly owned by this module
        for name, p in source._parameters.items():
            module._parameters[name] = (
                None
                if p is None
                else nn.Parameter(
                    p.to(device=device, dtype=dtype),
                    requires_grad=p.requires_grad,
                )
            )
        for name, b in source._buffers.items():
            module._buffers[name] = (
                None if b is None else b.to(device=device, dtype=dtype)
            )

    # Recurse to handle children
    for name, m in source._modules.items():
        module._modules[name] = (
            None if m is None else quantise(m, config, device, dtype, key + (name,))
        )
    return module


def test_quantise() -> None:
    for bits, tol, kernels in [
        (8, 0.01, ["triton", "torch.compile"]),
        (4, 0.2, ["triton", "torch.compile", "marlin", "marlin-lut"]),
        (2, 0.5, ["triton", "torch.compile"]),
        (1, 1, ["triton", "torch.compile"]),
    ]:
        for kernel in kernels:
            torch.manual_seed(100)
            device = torch.device("cuda")
            dtype = torch.float16 if kernel.startswith("marlin") else torch.bfloat16
            x = torch.randn((100, 384))
            ref_module = nn.Sequential(nn.Linear(384, 256, bias=False))
            module = quantise(
                ref_module,
                QuantisationConfig(bits=bits, block_size=64, kernel=kernel),
                device=device,
                dtype=dtype,
            )
            expected = ref_module(x)
            actual = module(x.to(device=device, dtype=dtype))
            assert actual.device.type == device.type, f"bad device {actual.device.type}"
            assert actual.dtype == dtype, f"bad dtype {actual.dtype}"
            qkernels._assert_rmsen(expected, actual.cpu(), tol=tol)

            # check stats, while we're here
            assert count_ops(module) == 2 * 384 * 256, "bad ops count"
            assert count_parameter_read_bytes(module) == (
                (256 * 384 * bits) // 8 + 256 * (384 // 64) * dtype.itemsize
            ), "bad bytes count"


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
        elif isinstance(module, marlin.Layer):
            ops += 2 * module.k * module.n
        else:
            ops += sum(2 * p.numel() for p in module.parameters(recurse=False))
    return ops


def count_parameter_read_bytes(model: transformers.PreTrainedModel) -> int:
    """Approximate bytes read per token."""
    bytes_r = 0
    for module in model.modules():
        if isinstance(module, marlin.Layer):
            # Like QuantisedLinear, don't count centroids (or workspace), only elements & scales
            bytes_r += sum(b.numel() * b.element_size() for b in [module.B, module.s])
        elif isinstance(module, nn.Embedding):
            pass  # Ignore sparse modules (embeddings), with negligible reads
        else:
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
    "12B": dict(
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
    "31B": dict(
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
            f"b={self.quantisation.block_size}"
            f" {self.quantisation.bits}-bit"
            f" {self.quantisation.kernel}"
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


def load_host_model(model_name: str) -> transformers.PreTrainedModel:
    if model_name.startswith("custom-llama-"):
        key = model_name.replace("custom-llama-", "")
        model = transformers.AutoModelForCausalLM.from_config(
            get_custom_llama_config(key), dtype=torch.float32
        )
        model.tokenizer = get_custom_llama_tokenizer()
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        )
        model.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model


def run_benchmark(
    settings: BenchmarkSettings,
    log: qbench.Log,
    host_model: transformers.PreTrainedModel,
) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.compiler.reset()
    torch._dynamo.config.recompile_limit = 64
    device = torch.device("cuda")
    if settings.quantisation is not None and settings.quantisation.kernel.startswith(
        "marlin"
    ):
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    out = dict(**settings.__dict__)
    if out["quantisation"] is not None:  # flatten
        out.update(out["quantisation"].__dict__)
    del out["quantisation"]

    try:
        if not _estimate_fits_on_device(settings, device, dtype):
            raise RuntimeError(
                "skipped; model size * bits/param is too large for the device"
            )

        out.update(parameters=sum(p.numel() for p in host_model.parameters()))
        model = quantise(host_model, settings.quantisation, device=device, dtype=dtype)
        result = benchmark(
            model, host_model.tokenizer, settings.batch_size, settings.reps
        )
        out.update(result.__dict__)
        print(f"{str(settings):>52}:  {result}", file=sys.stderr)

    except Exception as e:
        out.update(error=repr(e))
        print(f"{str(settings):>52}:  ERROR {e}", file=sys.stderr)

    log(out)


def _quantisation_configs(
    bits: list[int], block_sizes: list[int], kernels: list[str], skip: str
) -> Iterable[QuantisationConfig]:
    if 16 in bits:
        yield None  # 16-bit (no quantisation)
    for b in bits:
        if b < 16:
            for block_size in block_sizes:
                for kernel in kernels:
                    if not (kernel.startswith("marlin") and b != 4):
                        yield QuantisationConfig(
                            bits=b, block_size=block_size, kernel=kernel, skip=skip
                        )


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
        "--skip",
        default="^lm_head$",
        help=f"Regex of parameters to skip during quantisation",
    )
    parser.add_argument(
        "--block-size",
        default=[64],
        type=int,
        nargs="+",
        help="Quantisation block sizes (ignored for 16-bit)",
    )
    parser.add_argument(
        "--kernel",
        default=["triton", "marlin-lut"],
        choices=["triton", "torch.compile", "marlin", "marlin-lut"],
        type=str,
        nargs="+",
        help="Kernel implementation(s) to use (ignored for 16-bit)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=100,
        help="Number of generation steps to measure",
    )
    args = parser.parse_args()
    transformers.utils.logging.disable_progress_bar()

    test_quantise()

    with qbench.Log("models") as log:
        for model in args.model:
            host_model = load_host_model(model)
            for batch_size in args.batch_size:
                for config in _quantisation_configs(
                    args.bits, args.block_size, args.kernel, args.skip
                ):
                    settings = BenchmarkSettings(
                        model=model,
                        batch_size=batch_size,
                        quantisation=config,
                        reps=args.reps,
                    )
                    run_benchmark(settings, log, host_model)
            del host_model
            gc.collect()


if __name__ == "__main__":
    _main()
