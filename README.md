# Fused Dequantisation Kernels

This repository contains fused dequantisation kernels for various weight quantisation formats, along with benchmarking scripts to evaluate their performance on different models and batch sizes.

Commands to run benchmarks, first of kernels, then models:

```sh
python src/qbench.py
python src/qmodels.py

# Long runs
python src/qbench.py --exclude '' -b 16 8 4 2 1 -k 8192 6144 4096 3072
python src/qmodels.py --model custom-llama-4B custom-llama-12B custom-llama-31B --batch-size 1 4 16 64 256 --kernel triton marlin-lut marlin torch.compile
```

First-time setup:

```sh
sudo apt install ninja-build pybind11-dev
uv sync --extra dev
echo 'export PYTHONPATH=$(dirname ${VIRTUAL_ENV})/src' >> .venv/bin/activate
```

## Experimental

These benchmarks are not included the main benchmarks above and in our paper results, but are included here for reference.

**CPU (ARM Host)**

Requires `clang++` with {`openmp`, `libc++`}, and `ninja` on your PATH.

```sh
ninja -C src/experimental/cpu build/bench
./src/experimental/cpu/build/bench
```

**CPU (Android)**

Requires NDK at `/opt/android-sdk/ndk/latest` and `adb` on your PATH.

```sh
cd src/experimental/cpu
./run_on_device.sh
```

## Profiling

Requires `ncu`, [NVIDIA Nsight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started).

```sh
mkdir -p out/profiles
sudo $(which ncu) --kernel-name="regex:.*kernel__mv.*" --launch-skip=100 --launch-count=10 -o out/profiles/mv_lut8_4b $(which python) src/qbench.py --profile mv_lut8 -b 4
```

For experimental CPU kernels, we suggest inspecting the disassembly:

```sh
ninja -C src/experimental/cpu build/bench.s
# see src/experimental/cpu/build/bench.s
```

## Credits

Includes code from [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin), see [src/marlin/README.md](src/marlin/README.md) for details.

Our work was based on an Marlin port to add LUT support, written by [Sohir Maskey](https://github.com/SohirMaskey).

## License

Copyright (c) 2026 Graphcore Ltd. Licensed under the MIT License.

See LICENSE for further details.
