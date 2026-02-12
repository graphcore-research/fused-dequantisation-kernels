# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Fused dequantisation kernels and benchmarks."""

import importlib

__all__: list[str] = ["bench", "kernels", "marlin", "models"]


def __dir__():
    return sorted(list(globals().keys()) + __all__)


# Avoid importing submodules at package import time to prevent warnings when
# executing submodules with `python -m fdk.bench` (runpy loads the package
# first and later executes the submodule). Submodules are imported lazily on
# attribute access (PEP 562).
def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
