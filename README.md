# Quantisation benchmarking

```sh
python src/bench.py
```

Setup note: `echo 'export PYTHONPATH=$(dirname ${VIRTUAL_ENV})/src' >> .venv/bin/activate`

## Experimental code

```sh
# Build and run src/bench.cu
ninja -C src/experimental build/bench
./src/experimental/build/bench

# Note: build ptx
ninja -C src/experimental build/bench.ptx
```

## Profiling

```sh
mkdir -p out/profiles
sudo $(which ncu) --kernel-name="regex:.*kernel__mv.*" --launch-skip=100 --launch-count=10 -o out/profiles/k4_cuda_mv_4b_lut8 ./build/bench mv_4b_lut8 4096
sudo $(which ncu) --kernel-name="regex:.*kernel__mv.*" --launch-skip=100 --launch-count=10 -o out/profiles/k4_triton_mv_4b_lut8 $(which python) py/bench.py mv_4b_lut8 -k 4096
```

## Credits

Includes code from [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin), see [src/marlin/README.md](src/marlin/README.md) for details.
