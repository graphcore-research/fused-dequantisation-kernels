/**
 * Standalone benchmarks for dequantisation implemented in CUDA.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <variant>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cublasLt.h>

// --------------------------
// Utilities

constexpr int warp_size = 32;
using bfloat16 = __nv_bfloat16;
using bfloat162 = __nv_bfloat162;
struct half8 {
    half2 data[4];
};
struct bfloat168 {
    bfloat162 data[4];
};

template <class R, class T>
__device__ inline R reinterpret(T&& t) {
    static_assert(sizeof(R) == sizeof(T), "reinterpret size mismatch");
    return *reinterpret_cast<const R*>(&t);
}

// Lookup-table based 3-input logical operation (explicit as compiler doesn't generate in all cases)
template <uint immLut>
__device__ inline uint lop3(uint a, uint b, uint c) {
    uint d;
    asm volatile("lop3.b32 \t%0, %1, %2, %3, %4;" : "=r"(d) : "r"(a), "r"(b), "r"(c), "n"(immLut));
    return d;
}
constexpr uint LOP_A = 0xf0;
constexpr uint LOP_B = 0xcc;
constexpr uint LOP_C = 0xaa;

void check_cuda_error(const std::string& lib, int status, const std::string& file, int line) {
    if (status) {
        std::cerr << lib << " error at " << file << ":" << line << ": code=" << status;
        if (lib == "CUDA") {
            std::cerr << " :: " << cudaGetErrorString(static_cast<cudaError_t>(status));
        }
        std::cerr << std::endl;
        exit(1);
    }
}
#define CHECK_CUDA(call) check_cuda_error("CUDA", call, __FILE__, __LINE__);
#define CHECK_CUBLAS(call) check_cuda_error("cuBLAS", call, __FILE__, __LINE__);

struct Stream {
    cudaStream_t stream;
    Stream() { CHECK_CUDA(cudaStreamCreate(&stream)); }
    ~Stream() { cudaStreamDestroy(stream); }
    operator cudaStream_t() const { return stream; }
};

template <class T>
size_t count_bytes(std::initializer_list<thrust::device_vector<T>> vectors) {
    size_t total = 0;
    for (const auto& v : vectors) {
        total += v.size() * sizeof(T);
    }
    return total;
}

std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%dT%H-%M-%S");
    return oss.str();
}

std::string get_device_name() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    return prop.name;
}

std::string get_cuda_version() {
    int cuda_version;
    CHECK_CUDA(cudaRuntimeGetVersion(&cuda_version));
    uint cuda_major = cuda_version / 1000;
    uint cuda_minor = (cuda_version % 1000) / 10;
    return std::to_string(cuda_major) + "." + std::to_string(cuda_minor);
}

struct Log {
    using Value = std::variant<std::string, double, int64_t, uint64_t>;
    using Args = std::vector<std::tuple<std::string, Value>>;
    std::string id;
    std::ofstream file;
    std::vector<std::tuple<std::string, Value>> meta;

    explicit Log(Args meta_) : meta(meta_) {
        id = "cpp-" + get_current_timestamp();
        std::string log_dir = "out/dev";
        std::string log_path = log_dir + "/" + id + ".jsonl";
        if (system(("mkdir -p " + log_dir).c_str()) != 0) {
            std::cerr << "Failed to create log directory: " << log_dir << std::endl;
            exit(1);
        }
        file.open(log_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open log file: " << log_path << std::endl;
            throw std::runtime_error("Failed to open log file: " + log_path);
        }
        meta.push_back({"id", id});
    }

    void operator()(Args entries) {
        Args line = entries;
        line.insert(line.end(), meta.begin(), meta.end());
        file << "{";
        bool first = true;
        for (const auto& entry : line) {
            if (!first) {
                file << ",";
            }
            first = false;
            file << "\"" << std::get<0>(entry) << "\":";
            std::visit(
                [&](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, std::string>) {
                        file << "\"" << arg << "\"";
                    } else {
                        file << arg;
                    }
                },
                std::get<1>(entry));
        }
        file << "}" << std::endl;
    }
};

// --------------------------
// Baseline kernels

__global__ void kernel__copy(const uint4* in, uint4* out, size_t n) {
    const auto offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (auto i = offset; i < n; i += stride) {
        out[i] = in[i];
    }
}

__global__ void kernel__read_reduce_u32(const uint32_t* in, size_t n, unsigned long long* out_sum) {
    unsigned long long local_sum = 0;
    const auto offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = offset; i < n; i += stride) {
        local_sum += static_cast<unsigned long long>(in[i]);
    }

    // Reduce within block
    extern __shared__ unsigned long long shm[];
    shm[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // Atomic add across blocks
        atomicAdd(out_sum, shm[0]);
    }
}

__global__ void kernel__fill_u4(uint4* out, size_t n, uint4 value) {
    const auto offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = offset; i < n; i += stride) {
        out[i] = value;
    }
}

// --------------------------
// Conversion kernels

__global__ void kernel__copy_lut8(const uint4* in,
                                  const bfloat16* scale,
                                  const bfloat162* lut,
                                  bfloat16* out,
                                  size_t n,
                                  size_t group_size) {
    const auto offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    auto lookup = [lut](uint32_t idx, bfloat16 s) {
        auto v = __ldg(&lut[idx & 0xFF]);
        return reinterpret<uint32_t>(v * bfloat162{s, s});
    };
    auto lookup_x4 = [lookup](uint32_t v, bfloat16 s) {
        uint4 res;
        res.x = lookup(v >> 0, s);
        res.y = lookup(v >> 8, s);
        res.z = lookup(v >> 16, s);
        res.w = lookup(v >> 24, s);
        return res;
    };
    uint4* out_u4 = reinterpret_cast<uint4*>(out);
    for (auto i = offset; i < n; i += stride) {
        // Each iteration, read 4 * 4 = 16 bytes, and write 32 * 2-byte outputs
        auto v = in[i];
        auto s = scale[i / group_size];
        out_u4[4 * i + 0] = lookup_x4(v.x, s);
        out_u4[4 * i + 1] = lookup_x4(v.y, s);
        out_u4[4 * i + 2] = lookup_x4(v.z, s);
        out_u4[4 * i + 3] = lookup_x4(v.w, s);
    }
}

__global__ void kernel__copy_lut4(const uint4* in,
                                  const bfloat16* scale,
                                  const bfloat16* lut,
                                  bfloat16* out,
                                  size_t n,
                                  size_t group_size) {
    const auto offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    auto lookup = [lut](uint32_t idx, bfloat16 s) {
        auto v = __ldg(&lut[idx & 0xF]);
        return static_cast<uint32_t>(reinterpret<uint16_t>(v * s));
    };
    auto lookup_x8 = [lookup](uint32_t v, bfloat16 s) {
        uint4 res;
        res.x = (lookup(v >> 0, s) << 16) | lookup(v >> 4, s);
        res.y = (lookup(v >> 8, s) << 16) | lookup(v >> 12, s);
        res.z = (lookup(v >> 16, s) << 16) | lookup(v >> 20, s);
        res.w = (lookup(v >> 24, s) << 16) | lookup(v >> 28, s);
        return res;
    };
    uint4* out_u4 = reinterpret_cast<uint4*>(out);
    for (auto i = offset; i < n; i += stride) {
        // Each iteration, read 4 * 4 = 16 bytes, and write 32 * 2-byte outputs
        auto v = in[i];
        auto s = scale[i / group_size];
        out_u4[4 * i + 0] = lookup_x8(v.x, s);
        out_u4[4 * i + 1] = lookup_x8(v.y, s);
        out_u4[4 * i + 2] = lookup_x8(v.z, s);
        out_u4[4 * i + 3] = lookup_x8(v.w, s);
    }
}

__device__ inline half8 dequant_linear4_fp16(uint q, half s) {
    const auto MASK = 0x64006400u;
    const auto a = reinterpret<half2>(0x64086408);  // {1032, 1032}
    const auto b = reinterpret<half2>(0x2c002c00);  // {1/16, 1/16}
    const auto c = reinterpret<half2>(0xd480d480);  // {-72, -72}

    auto v04 = lop3<(LOP_A & LOP_B) | LOP_C>(q, 0x000f000f, MASK);
    auto v15 = lop3<(LOP_A & LOP_B) | LOP_C>(q, 0x00f000f0, MASK);
    auto qs = q >> 8;
    auto v26 = lop3<(LOP_A & LOP_B) | LOP_C>(qs, 0x000f000f, MASK);
    auto v37 = lop3<(LOP_A & LOP_B) | LOP_C>(qs, 0x00f000f0, MASK);

    half8 out;
    auto s2 = half2{s, s};
    out.data[0] = s2 * (reinterpret<half2>(v04) - a);
    out.data[1] = s2 * __hfma2(reinterpret<half2>(v15), b, c);
    out.data[2] = s2 * (reinterpret<half2>(v26) - a);
    out.data[3] = s2 * __hfma2(reinterpret<half2>(v37), b, c);
    return out;
}

__global__ void kernel__copy_linear4_fp16(const uint4* in,
                                          const half* scale,
                                          half* out,
                                          size_t n,
                                          size_t group_size) {
    const auto offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    uint4* out_u4 = reinterpret_cast<uint4*>(out);
    for (auto i = offset; i < n; i += stride) {
        // Each iteration, read 4 * 4 = 16 bytes, and write 32 * fp16 outputs
        auto v = in[i];
        auto s = scale[i / group_size];
        out_u4[4 * i + 0] = reinterpret<uint4>(dequant_linear4_fp16(v.x, s).data);
        out_u4[4 * i + 1] = reinterpret<uint4>(dequant_linear4_fp16(v.y, s).data);
        out_u4[4 * i + 2] = reinterpret<uint4>(dequant_linear4_fp16(v.z, s).data);
        out_u4[4 * i + 3] = reinterpret<uint4>(dequant_linear4_fp16(v.w, s).data);
    }
}

__device__ inline bfloat168 dequant_binary(uint8_t q, bfloat16 s) {
    uint16_t res[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        res[i] = reinterpret<uint16_t>(-s);
        res[i] ^= (uint16_t(q) << (8 + i)) & 0x8000;
    }
    return reinterpret<bfloat168>(res);
}

__global__ void kernel__copy_binary(const uint32_t* in,
                                    const bfloat16* scale,
                                    bfloat16* out,
                                    size_t n,
                                    size_t group_size) {
    const auto offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    uint4* out_u4 = reinterpret_cast<uint4*>(out);
    for (auto i = offset; i < n; i += stride) {
        // Each iteration, read 4 bytes, and write 32 * 2-byte outputs
        auto v = in[i];
        auto s = scale[i / group_size];
        // little-endian, and we want to read MSB first
        out_u4[4 * i + 0] = reinterpret<uint4>(dequant_binary((v >> 24) & 0xFF, s));
        out_u4[4 * i + 1] = reinterpret<uint4>(dequant_binary((v >> 16) & 0xFF, s));
        out_u4[4 * i + 2] = reinterpret<uint4>(dequant_binary((v >> 8) & 0xFF, s));
        out_u4[4 * i + 3] = reinterpret<uint4>(dequant_binary((v >> 0) & 0xFF, s));
    }
}

// --------------------------
// Matrix-vector kernels

__global__ void kernel__mv(const bfloat16* a,  // [k]
                           const bfloat16* b,  // [k, n] (k-major)
                           bfloat16* out,      // [n]
                           int k,
                           int n) {
    extern __shared__ char _shared[];
    auto shared = reinterpret_cast<float2*>(_shared);
    const auto in2 = blockIdx.x * blockDim.x + threadIdx.x;
    const auto a2 = reinterpret_cast<const bfloat162*>(a);
    const auto b2 = reinterpret_cast<const bfloat162*>(b);
    const auto out2 = reinterpret_cast<bfloat162*>(out);
    if (in2 < n / 2) {
        // Partial dot product for outputs n = [2*in2, 2*in2+1]
        // with partial sums spread across threadIdx.y
        float2 sum = {0.0f, 0.0f};
        for (auto ik2 = threadIdx.y; ik2 < k / 2; ik2 += blockDim.y) {
            auto a_ik = __bfloat1622float2(a2[ik2]);
            auto b_ik0 = __bfloat1622float2(b2[(2 * ik2) * n / 2 + in2]);
            auto b_ik1 = __bfloat1622float2(b2[(2 * ik2 + 1) * n / 2 + in2]);
            sum.x = fmaf(a_ik.x, b_ik0.x, sum.x);  // n = 2*in2, k = 2*ik2
            sum.x = fmaf(a_ik.y, b_ik1.x, sum.x);  // n = 2*in2, k = 2*ik2+1
            sum.y = fmaf(a_ik.x, b_ik0.y, sum.y);  // n = 2*in2+1, k = 2*ik2
            sum.y = fmaf(a_ik.y, b_ik1.y, sum.y);  // n = 2*in2+1, k = 2*ik2+1
        }
        // Reduce over threadIdx.y in shared memory
        auto sbase = threadIdx.x * blockDim.y;
        shared[sbase + threadIdx.y] = sum;
        __syncthreads();
        if (threadIdx.y == 0) {
            // A linear reduction is fine - this isn't the bottleneck
            for (auto i = 1u; i < blockDim.y; ++i) {
                sum.x += shared[sbase + i].x;
                sum.y += shared[sbase + i].y;
            }
            out2[in2] = __float22bfloat162_rn(sum);
        }
    }
}

void run_mv(const bfloat16* a, const bfloat16* b, bfloat16* out, int k, int n) {
    if (n % 2 != 0 || k % 2 != 0) {
        throw std::invalid_argument("kernel__mv requires n and k to be even");
    }
    dim3 blockDim(32, 32);  // [bn, bk]
    dim3 gridDim((n + blockDim.x - 1) / (2 * blockDim.x), 1);
    auto sharedBytes = blockDim.y * blockDim.x * sizeof(float2);
    kernel__mv<<<gridDim, blockDim, sharedBytes>>>(a, b, out, k, n);
    CHECK_CUDA(cudaGetLastError());
}

template <class T>
__device__ T block_reduce(T value, T* shared) {
    // Reduce within a warp using shuffle
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    // Reduce across warps in shared memory
    if ((threadIdx.x % warp_size) == 0) {
        shared[threadIdx.x / warp_size] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // A linear reduction is fine - this isn't the bottleneck
        for (auto i = 1u; i < blockDim.x / warp_size; ++i) {
            value += shared[i];
        }
    }
    return value;
}

__global__ void kernel__mvT(const bfloat16* a,  // [k]
                            const bfloat16* b,  // [n, k] (n-major)
                            bfloat16* out,      // [n]
                            int k,
                            int n) {
    extern __shared__ char _shared[];
    const auto a2 = reinterpret_cast<const bfloat162*>(a);
    const auto b2 = reinterpret_cast<const bfloat162*>(b);
    const auto in = blockIdx.x;
    // Partial dot product with sums spread across threadIdx.x
    float sum = 0.0f;
    for (auto ik2 = threadIdx.x; ik2 < k / 2; ik2 += blockDim.x) {
        auto a_ik = __bfloat1622float2(a2[ik2]);
        auto b_in = __bfloat1622float2(b2[in * k / 2 + ik2]);
        sum = fmaf(a_ik.x, b_in.x, sum);  // n = in, k = 2*ik2
        sum = fmaf(a_ik.y, b_in.y, sum);  // n = in, k = 2*ik2+1
    }
    sum = block_reduce(sum, reinterpret_cast<float*>(_shared));
    if (threadIdx.x == 0) {
        out[in] = __float2bfloat16_rn(sum);
    }
}

void run_mvT(const bfloat16* a, const bfloat16* b, bfloat16* out, int k, int n) {
    if (k % 2 != 0) {
        throw std::invalid_argument("kernel__mvT requires k to be even");
    }
    auto blockSize = 512;
    auto sharedBytes = (blockSize / warp_size) * sizeof(float);
    kernel__mvT<<<n, blockSize, sharedBytes>>>(a, b, out, k, n);
    CHECK_CUDA(cudaGetLastError());
}

// --------------------------
// Testing

template <class T>
void expect_eq(const T& a, const T& b, const std::string& msg) {
    if (a != b) {
        std::ostringstream oss;
        oss << "Error, expected: " << a << ", actual: " << b << ",  " << msg;
        throw std::runtime_error(oss.str());
    }
}

template <class T>
void expect_eq(const T& expected, const T& actual, const std::string& msg, T tolerance) {
    if (std::abs(expected - actual) > tolerance) {
        std::ostringstream oss;
        oss << "Error, expected: " << expected << ", actual: " << actual << " (tolerance "
            << tolerance << "),  " << msg;
        throw std::runtime_error(oss.str());
    }
}

void test_lut() {
    // Problem
    thrust::device_vector<bfloat16> d_lut4(16);
    for (int i = 0; i < 16; i++) {
        d_lut4[i] = __float2bfloat16(i - 8);
    }
    thrust::device_vector<uint8_t> d_in({0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF});
    thrust::host_vector<float> expected({-8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
                                         0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    thrust::device_vector<bfloat16> d_scale = {1.0f, 128.0f, 1 / 256.0f, 1.0f};
    const size_t group_size = 64;
    const size_t n = d_scale.size() * group_size;

    // Expand {d_in, expected} to be of size n
    assert(n % expected.size() == 0);
    const auto d_in_initial = d_in;
    const auto expected_initial = expected;
    while (expected.size() < n) {
        d_in.insert(d_in.end(), d_in_initial.begin(), d_in_initial.end());
        expected.insert(expected.end(), expected_initial.begin(), expected_initial.end());
    }
    // Apply scale
    for (size_t i = 0; i < expected.size(); i++) {
        expected[i] *= __bfloat162float(d_scale[i / group_size]);
    }

    // 4-bit LUT
    {
        thrust::device_vector<bfloat16> d_out(n);
        const float sentinel = 1.96875f;
        d_out.push_back(sentinel);
        kernel__copy_lut4<<<3, 2>>>(reinterpret_cast<const uint4*>(d_in.data().get()),
                                    d_scale.data().get(), d_lut4.data().get(), d_out.data().get(),
                                    n / 32, group_size / 32);
        CHECK_CUDA(cudaGetLastError());
        expect_eq(sentinel, __bfloat162float(d_out.back()), "sentinel check");
        for (size_t i = 0; i < n; i++) {
            float v = __bfloat162float(d_out[i]);
            expect_eq(expected[i], v, "at index " + std::to_string(i), 1e-3f);
        }
    }

    // 8-bit LUT
    {
        thrust::host_vector<bfloat162> h_lut8(256);
        for (int i = 0; i < 256; i++) {
            h_lut8[i] = bfloat162(d_lut4[i / 16], d_lut4[i % 16]);
        }
        thrust::device_vector<bfloat162> d_lut8 = h_lut8;

        thrust::device_vector<bfloat16> d_out(n);
        const float sentinel = 1.96875f;
        d_out.push_back(sentinel);
        kernel__copy_lut8<<<3, 2>>>(reinterpret_cast<const uint4*>(d_in.data().get()),
                                    d_scale.data().get(), d_lut8.data().get(), d_out.data().get(),
                                    n / 32, group_size / 32);
        CHECK_CUDA(cudaGetLastError());
        expect_eq(sentinel, __bfloat162float(d_out.back()), "sentinel check");
        for (size_t i = 0; i < n; i++) {
            float v = __bfloat162float(d_out[i]);
            expect_eq(v, expected[i], "at index " + std::to_string(i), 1e-3f);
        }
    }
}

void test_linear4_fp16() {
    // Problem
    thrust::device_vector<uint8_t> d_in({0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF});
    // Order defined by the shuffle in dequant_linear4_fp16
    thrust::host_vector<float> expected({-7, -3, -8, -4, -5, -1, -6, -2, 1, 5, 0, 4, 3, 7, 2, 6});
    thrust::device_vector<half> d_scale = {1.0f, 128.0f, 1 / 256.0f, 1.0f};
    const size_t group_size = 64;
    const size_t n = d_scale.size() * group_size;

    // Expand {d_in, expected} to be of size n
    assert(n % expected.size() == 0);
    const auto d_in_initial = d_in;
    const auto expected_initial = expected;
    while (expected.size() < n) {
        d_in.insert(d_in.end(), d_in_initial.begin(), d_in_initial.end());
        expected.insert(expected.end(), expected_initial.begin(), expected_initial.end());
    }
    // Apply scale
    for (size_t i = 0; i < expected.size(); i++) {
        expected[i] *= __half2float(d_scale[i / group_size]);
    }

    // Test
    thrust::device_vector<half> d_out(n);
    const float sentinel = 1.96875f;
    d_out.push_back(sentinel);
    kernel__copy_linear4_fp16<<<3, 2>>>(reinterpret_cast<const uint4*>(d_in.data().get()),
                                        d_scale.data().get(), d_out.data().get(), n / 32,
                                        group_size / 32);
    CHECK_CUDA(cudaGetLastError());
    expect_eq(sentinel, __half2float(d_out.back()), "sentinel check");
    for (size_t i = 0; i < n; i++) {
        float v = __half2float(d_out[i]);
        expect_eq(v, expected[i], "at index " + std::to_string(i), 1e-3f);
    }
}

void test_binary() {
    // Problem
    thrust::device_vector<uint32_t> d_in({0b11100000'10101010'00000000'11111111, 0xf1234567});
    thrust::device_vector<bfloat16> d_scale = {3.0f, 0.5f, 1.0f, 2.0f};
    const size_t group_size = 64;
    assert(32 * d_in.size() == group_size);
    const size_t n = group_size * d_scale.size();

    // Build expected output
    thrust::host_vector<float> expected;
    for (auto i = 0u; i < d_in.size(); i++) {
        for (int b = 31; b >= 0; b--) {
            expected.push_back((d_in[i] & (1u << b)) ? 1 : -1);
        }
    }
    // Expand {d_in, expected} to be of size n
    assert(n % expected.size() == 0);
    const auto d_in_initial = d_in;
    const auto expected_initial = expected;
    while (expected.size() < n) {
        d_in.insert(d_in.end(), d_in_initial.begin(), d_in_initial.end());
        expected.insert(expected.end(), expected_initial.begin(), expected_initial.end());
    }
    // Apply scale
    for (size_t i = 0; i < expected.size(); i++) {
        expected[i] *= __bfloat162float(d_scale[i / group_size]);
    }

    // Test
    thrust::device_vector<bfloat16> d_out(n);
    const float sentinel = 1.96875f;
    d_out.push_back(sentinel);
    kernel__copy_binary<<<3, 2>>>(d_in.data().get(), d_scale.data().get(), d_out.data().get(),
                                  n / 32, group_size / 32);
    CHECK_CUDA(cudaGetLastError());
    expect_eq(sentinel, __bfloat162float(d_out.back()), "sentinel check");
    for (size_t i = 0; i < n; i++) {
        expect_eq(expected[i], __bfloat162float(d_out[i]), "at index " + std::to_string(i), 1e-3f);
    }
}

void test_mv() {
    const int k = 2300;
    const int n = 180;
    thrust::host_vector<bfloat16> h_a(k);
    thrust::host_vector<bfloat16> h_b(k * n);
    std::default_random_engine rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < k; i++) {
        h_a[i] = __float2bfloat16(dist(rng));
    }
    for (int i = 0; i < k * n; i++) {
        h_b[i] = __float2bfloat16(dist(rng));
    }
    thrust::host_vector<bfloat16> expected(n);
    thrust::host_vector<bfloat16> expectedT(n);
    for (int j = 0; j < n; j++) {
        float sum = 0.0f, sumT = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += __bfloat162float(h_a[i]) * __bfloat162float(h_b[i * n + j]);
            sumT += __bfloat162float(h_a[i]) * __bfloat162float(h_b[j * k + i]);
        }
        expected[j] = __float2bfloat16(sum);
        expectedT[j] = __float2bfloat16(sumT);
    }

    {
        thrust::device_vector<bfloat16> d_a = h_a;
        thrust::device_vector<bfloat16> d_b = h_b;
        thrust::device_vector<bfloat16> d_out(n);
        run_mv(d_a.data().get(), d_b.data().get(), d_out.data().get(), k, n);
        thrust::host_vector<bfloat16> h_out = d_out;

        for (int i = 0; i < n; i++) {
            expect_eq(__bfloat162float(expected[i]), __bfloat162float(h_out[i]),
                      "mv output at index " + std::to_string(i), 1e-3f);
        }
    }

    {
        thrust::device_vector<bfloat16> d_a = h_a;
        thrust::device_vector<bfloat16> d_b = h_b;
        thrust::device_vector<bfloat16> d_out(n);
        run_mvT(d_a.data().get(), d_b.data().get(), d_out.data().get(), k, n);
        thrust::host_vector<bfloat16> h_out = d_out;

        for (int i = 0; i < n; i++) {
            expect_eq(__bfloat162float(expectedT[i]), __bfloat162float(h_out[i]),
                      "mvT output at index " + std::to_string(i), 1e-3f);
        }
    }
}

// --------------------------
// Benchmarking

double measure_time(uint reps, cudaStream_t stream, const std::function<void(uint)>& fn) {
    for (uint i = 0; i < reps; i++) {
        fn(i);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (uint i = 0; i < reps; i++) {
        fn(i);
    }
    CHECK_CUDA(cudaEventRecord(end, stream));
    CHECK_CUDA(cudaEventSynchronize(end));

    float time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, end));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));

    return time_ms * 1e-3 / reps;
}

void benchmark_conversions(Log& log) {
    constexpr size_t bytes = 4 * (1ull << 30);
    static_assert(bytes % 16 == 0, "bytes must be a multiple of 16 for uint4 fill");
    const uint reps = 10;
    std::regex pattern("^(.*)$");

    Stream stream;
    thrust::device_vector<uint8_t> d_a(bytes);
    thrust::device_vector<uint8_t> d_b(bytes);
    thrust::sequence(thrust::cuda::par.on(stream), d_a.begin(), d_a.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), d_b.begin(), d_b.end(), 0);

    auto run_benchmark = [&](const std::string& name, size_t bytes_read, size_t bytes_write,
                             const std::function<void(uint)>& fn) {
        if (!std::regex_match(name, pattern)) {
            return;
        }
        double avg_time = measure_time(reps, stream, fn);
        std::ostringstream time_str, bw_str;
        time_str << std::fixed << std::setprecision(2) << (avg_time * 1e3) << " ms";
        bw_str << std::fixed << std::setprecision(2)
               << ((bytes_read + bytes_write) / avg_time) / 1e9 << " GB/s";
        std::cerr << std::setw(18) << std::left << name << "  " << std::setw(12) << std::right
                  << time_str.str() << "  " << std::setw(12) << std::right << bw_str.str()
                  << std::endl;

        // Write JSON log (one object per measurement)
        log({{"test", name},
             {"avg_time", avg_time},
             {"bytes_read", static_cast<uint64_t>(bytes_read)},
             {"bytes_write", static_cast<uint64_t>(bytes_write)},
             {"reps", static_cast<uint64_t>(reps)}});
    };

    std::cerr << "### benchmark_conversions" << std::endl;

    // Baseline measurements

    run_benchmark("cudamemcpy", bytes, bytes, [&](uint) {
        CHECK_CUDA(cudaMemcpyAsync(d_b.data().get(), d_a.data().get(), bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    });

    run_benchmark("copy", bytes, bytes, [&](uint) {
        int threads = 1024;
        int blocks = 65536;
        const size_t n4 = bytes / sizeof(uint4);
        kernel__copy<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint4*>(d_a.data().get()),
            reinterpret_cast<uint4*>(d_b.data().get()), n4);
        CHECK_CUDA(cudaGetLastError());
    });

    thrust::device_vector<unsigned long long> d_sum(1);
    run_benchmark("read_reduce_sum", bytes, sizeof(unsigned long long), [&](uint) {
        CHECK_CUDA(cudaMemsetAsync(d_sum.data().get(), 0, sizeof(unsigned long long), stream));
        int threads = 1024;
        int blocks = 16384;
        const size_t n = bytes / sizeof(uint32_t);
        kernel__read_reduce_u32<<<blocks, threads, threads * sizeof(unsigned long long), stream>>>(
            reinterpret_cast<const uint32_t*>(d_a.data().get()), n, d_sum.data().get());
        CHECK_CUDA(cudaGetLastError());
    });

    run_benchmark("write_fill", sizeof(uint4), bytes, [&](uint) {
        int threads = 1024;
        int blocks = 65536;  // since we have a lot of memory to fill
        const uint32_t v = 0x12345678u;
        const size_t n4 = bytes / sizeof(uint4);
        kernel__fill_u4<<<blocks, threads, 0, stream>>>(reinterpret_cast<uint4*>(d_b.data().get()),
                                                        n4, uint4{v, v, v, v});
        CHECK_CUDA(cudaGetLastError());
    });

    // Main measurements

    // LUT8
    {
        const auto n = bytes / sizeof(bfloat16);
        const auto g = 64;
        const auto bytes_read = n / 2 + n / g * sizeof(bfloat16);
        const auto bytes_write = n * sizeof(bfloat16);

        thrust::device_vector<bfloat16> d_b16(n);
        thrust::device_vector<bfloat16> d_scale(n / g, 0.01f);
        thrust::host_vector<bfloat162> h_lut8(256);
        for (int i = 0; i < 256; i++) {
            h_lut8[i] = bfloat162{i % 16, i / 16};
        }
        thrust::device_vector<bfloat162> d_lut8(256);
        thrust::copy(h_lut8.begin(), h_lut8.end(), d_lut8.begin());
        run_benchmark("copy_lut8", bytes_read, bytes_write, [&](uint) {
            int threads = 1024;
            int blocks = 65536;
            kernel__copy_lut8<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_a.data().get()), d_scale.data().get(),
                d_lut8.data().get(), reinterpret_cast<bfloat16*>(d_b16.data().get()), n / 32,
                g / 32);
            CHECK_CUDA(cudaGetLastError());
        });
    }

    // LUT4
    {
        const auto n = bytes / sizeof(bfloat16);
        const auto g = 64;
        const auto bytes_read = n / 2 + n / g * sizeof(bfloat16);
        const auto bytes_write = n * sizeof(bfloat16);

        thrust::device_vector<bfloat16> d_b16(n);
        thrust::device_vector<bfloat16> d_scale(n / g, 0.01f);
        thrust::host_vector<bfloat16> h_lut4(16);
        for (int i = 0; i < 16; i++) {
            h_lut4[i] = bfloat16(i);
        }
        thrust::device_vector<bfloat16> d_lut4(16);
        thrust::copy(h_lut4.begin(), h_lut4.end(), d_lut4.begin());
        run_benchmark("copy_lut4", bytes_read, bytes_write, [&](uint) {
            int threads = 1024;
            int blocks = 65536;
            kernel__copy_lut4<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_a.data().get()), d_scale.data().get(),
                d_lut4.data().get(), reinterpret_cast<bfloat16*>(d_b16.data().get()), n / 32,
                g / 32);
            CHECK_CUDA(cudaGetLastError());
        });
    }

    // Linear4 fp16
    {
        const auto n = bytes / sizeof(half);
        const auto g = 64;
        const auto bytes_read = n / 2 + n / g * sizeof(half);
        const auto bytes_write = n * sizeof(half);

        thrust::device_vector<half> d_fp16(n);
        thrust::device_vector<half> d_scale(n / g, 0.01f);
        run_benchmark("copy_linear4_fp16", bytes_read, bytes_write, [&](uint) {
            int threads = 1024;
            int blocks = 65536;
            kernel__copy_linear4_fp16<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_a.data().get()), d_scale.data().get(),
                d_fp16.data().get(), n / 32, g / 32);
            CHECK_CUDA(cudaGetLastError());
        });
    }

    // Binary
    {
        const auto n = bytes / sizeof(bfloat16);
        const auto g = 64;
        const auto bytes_read = n / 8 + n / g * sizeof(bfloat16);
        const auto bytes_write = n * sizeof(bfloat16);

        thrust::device_vector<bfloat16> d_b16(n);
        thrust::device_vector<bfloat16> d_scale(n / g, 0.01f);
        run_benchmark("copy_binary", bytes_read, bytes_write, [&](uint) {
            int threads = 1024;
            int blocks = 65536;
            kernel__copy_binary<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint32_t*>(d_a.data().get()), d_scale.data().get(),
                d_b16.data().get(), n / 32, g / 32);
            CHECK_CUDA(cudaGetLastError());
        });
    }

    std::cerr << std::endl;
}

void benchmark_mv(Log& log) {
    const int k = 8192;
    const int n = 4096;
    const int reps = 100;
    const int copies = 10;  // cycle through separate args to avoid cache hits

    Stream stream;

    auto run_benchmark = [&](const std::string& name, size_t bytes_read, size_t bytes_write,
                             const std::function<void(uint)>& fn) {
        double avg_time = measure_time(reps, stream, fn);
        std::ostringstream time_str, bw_str;
        time_str << std::fixed << std::setprecision(2) << (avg_time * 1e6) << " us";
        bw_str << std::fixed << std::setprecision(2)
               << ((bytes_read + bytes_write) / avg_time) / 1e9 << " GB/s";
        std::cerr << std::setw(18) << std::left << name << "  " << std::setw(12) << std::right
                  << time_str.str() << "  " << std::setw(12) << std::right << bw_str.str()
                  << std::endl;

        // Write JSON log (one object per measurement)
        log({{"test", name},
             {"avg_time", avg_time},
             {"bytes_read", static_cast<uint64_t>(bytes_read)},
             {"bytes_write", static_cast<uint64_t>(bytes_write)},
             {"reps", static_cast<uint64_t>(reps)}});
    };

    std::cerr << "### benchmark_mv" << std::endl;

    // cublaslt_mv
    {
        thrust::device_vector<bfloat16> d_a(copies * k, 1.5f);
        thrust::device_vector<bfloat16> d_b(copies * k * n, 0.125f);
        thrust::device_vector<bfloat16> d_out(copies * n);
        float alpha = 1.0f;
        float beta = 0.0f;

        // "Light" API setup
        cublasLtHandle_t ltHandle;
        CHECK_CUBLAS(cublasLtCreate(&ltHandle));
        cublasLtMatmulDesc_t operationDesc = nullptr;
        cublasLtMatrixLayout_t adesc = nullptr, bdesc = nullptr, cdesc = nullptr;
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&adesc, CUDA_R_16BF, k, 1, k));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_16BF, n, k, n));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_16BF, n, 1, n));

        run_benchmark(
            "cublaslt_mv", (k + k * n) * sizeof(bfloat16), n * sizeof(bfloat16), [&](uint i) {
                auto idx = i % copies;
                CHECK_CUBLAS(cublasLtMatmul(
                    ltHandle, operationDesc, &alpha, d_b.data().get() + idx * k * n, bdesc,
                    d_a.data().get() + idx * k, adesc, &beta, d_out.data().get() + idx * n, cdesc,
                    d_out.data().get() + idx * n, cdesc, nullptr, nullptr, 0, stream));
            });

        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(adesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(bdesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(cdesc));
        CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
        CHECK_CUBLAS(cublasLtDestroy(ltHandle));
    }

    // mv
    {
        thrust::device_vector<bfloat16> d_a(copies * k, 1.5f);
        thrust::device_vector<bfloat16> d_b(copies * k * n, 0.125f);
        thrust::device_vector<bfloat16> d_out(copies * n);
        run_benchmark("mv", (k + k * n) * sizeof(bfloat16), n * sizeof(bfloat16), [&](uint i) {
            auto idx = i % copies;
            run_mv(d_a.data().get() + idx * k, d_b.data().get() + idx * k * n,
                   d_out.data().get() + idx * n, k, n);
        });
    }

    // mvT
    {
        thrust::device_vector<bfloat16> d_a(copies * k, 1.5f);
        thrust::device_vector<bfloat16> d_b(copies * n * k, 0.125f);
        thrust::device_vector<bfloat16> d_out(copies * n);
        run_benchmark("mvT", (k + n * k) * sizeof(bfloat16), n * sizeof(bfloat16), [&](uint i) {
            auto idx = i % copies;
            run_mvT(d_a.data().get() + idx * k, d_b.data().get() + idx * n * k,
                    d_out.data().get() + idx * n, k, n);
        });
    }

    std::cerr << std::endl;
}

// --------------------------
// Driver program

int main() {
    auto device_name = get_device_name();
    auto cuda_version = get_cuda_version();
    std::cerr << "Using device: " << device_name << std::endl;
    std::cerr << "CUDA version: " << cuda_version << std::endl;
    std::cerr << std::endl;

    // Tests
    bool test_passed = true;
    for (auto& test : {test_lut, test_linear4_fp16, test_binary, test_mv}) {
        try {
            test();
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            test_passed = false;
            continue;
        }
    }
    if (!test_passed) {
        std::cerr << "!!! TESTS FAILED !!!\n" << std::endl;
    }

    // Benchmarking
    Log log({{"device", device_name}, {"cuda_version", cuda_version}});
    benchmark_conversions(log);
    benchmark_mv(log);

    return 0;
}
