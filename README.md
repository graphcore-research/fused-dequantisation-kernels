# Quantisation benchmarking

```sh
# Build and run cpp/bench.cu
ninja build/bench
./build/bench

# Note: build ptx
ninja build/bench.ptx
```

## Profiling

```sh
mkdir -p out/profiles
sudo $(which ncu) --kernel-name="regex:.*kernel__mv.*" --launch-skip=100 --launch-count=10 -o out/profiles/k4_cuda_mv_4b_lut8 ./build/bench mv_4b_lut8 4096
sudo $(which ncu) --kernel-name="regex:.*kernel__mv.*" --launch-skip=100 --launch-count=10 -o out/profiles/k4_triton_mv_4b_lut8 $(which python) py/bench.py mv_4b_lut8 -k 4096
```
