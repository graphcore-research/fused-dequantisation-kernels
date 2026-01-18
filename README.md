# Quantisation benchmarking

```sh
python src/qbench.py
python src/qmodels.py

# Long runs
python src/qbench.py --exclude '' -b 16 8 4 2 1 -k 8192 6144 4096 3072
python src/qmodels.py --model custom-llama-4B custom-llama-11B custom-llama-30B --batch-size 1 4 16 64
```

First-time setup:

```sh
sudo apt install ninja-build pybind11-dev
uv sync
echo 'export PYTHONPATH=$(dirname ${VIRTUAL_ENV})/src' >> .venv/bin/activate
```

## Experimental code

```sh
# Build and run src/experimental/bench.cu
ninja -C src/experimental build/bench
./src/experimental/build/bench

# Note: build ptx
ninja -C src/experimental build/bench.ptx
```

## Profiling

Requires `ncu`, [NVIDIA Nsight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started).

```sh
mkdir -p out/profiles
sudo $(which ncu) --kernel-name="regex:.*kernel__mv.*" --launch-skip=100 --launch-count=10 -o out/profiles/cuda_mv_lut8_4b ./src/experimental/build/bench mv_4b_lut8 4096
sudo $(which ncu) --kernel-name="regex:.*kernel__mv.*" --launch-skip=100 --launch-count=10 -o out/profiles/mv_lut8_4b $(which python) src/qbench.py --profile mv_lut8 -b 4
```

## Credits

Includes code from [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin), see [src/marlin/README.md](src/marlin/README.md) for details.
