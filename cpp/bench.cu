/**
 * Standalone benchmark for CUDA transfer bandwidth
 * (read-only, write-only, read + write = memcpy).
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <variant>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

// --------------------------
// Utilities

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

struct Stream {
    cudaStream_t stream;
    Stream() { CHECK_CUDA(cudaStreamCreate(&stream)); }
    ~Stream() { cudaStreamDestroy(stream); }
    operator cudaStream_t() const { return stream; }
};

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
    std::string id;
    std::ofstream file;

    Log() {
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
    }

    void operator()(
        std::initializer_list<
            std::pair<std::string, std::variant<std::string, double, int64_t, uint64_t>>> entries) {
        file << "{";
        bool first = true;
        for (const auto& entry : entries) {
            if (!first) {
                file << ",";
            }
            first = false;
            file << "\"" << entry.first << "\":";
            std::visit(
                [&](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, std::string>) {
                        file << "\"" << arg << "\"";
                    } else {
                        file << arg;
                    }
                },
                entry.second);
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
// Main kernels

template <class R, class T>
__device__ inline R reinterpret(T&& t) {
    return *reinterpret_cast<const R*>(&t);
}

template <class T>
__global__ void kernel__copy_lut8(const uint4* in, const T* lut, T* out, size_t n) {
    static_assert(sizeof(T) == 4, "T must be 4 bytes (e.g., __nv_bfloat162)");
    const auto offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const uint32_t* lut_u = reinterpret_cast<const uint32_t*>(lut);
    auto lookup_x4 = [lut_u](uint32_t v) {
        uint4 res;
        res.x = __ldg(&lut_u[(v >> 0) & 0xFF]);
        res.y = __ldg(&lut_u[(v >> 8) & 0xFF]);
        res.z = __ldg(&lut_u[(v >> 16) & 0xFF]);
        res.w = __ldg(&lut_u[(v >> 24) & 0xFF]);
        return res;
    };
    uint4* out_u4 = reinterpret_cast<uint4*>(out);
    for (auto i = offset; i < n; i += stride) {
        // Each iteration, read 4 * 4 = 16 bytes, and write 16 * 4-byte outputs
        auto v = in[i];
        out_u4[4 * i + 0] = lookup_x4(v.x);
        out_u4[4 * i + 1] = lookup_x4(v.y);
        out_u4[4 * i + 2] = lookup_x4(v.z);
        out_u4[4 * i + 3] = lookup_x4(v.w);
    }
}

template <class T>
__global__ void kernel__copy_lut4(const uint4* in, const T* lut, T* out, size_t n) {
    static_assert(sizeof(T) == 2, "T must be 2 bytes (e.g., __nv_bfloat16)");
    const auto offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const uint16_t* lut_u = reinterpret_cast<const uint16_t*>(lut);
    auto lookup_x8 = [lut_u](uint32_t v) {
        uint4 res;
        res.x = (static_cast<uint32_t>(__ldg(&lut_u[(v >> 0) & 0xF])) << 16) |
                (static_cast<uint32_t>(__ldg(&lut_u[(v >> 4) & 0xF])));
        res.y = (static_cast<uint32_t>(__ldg(&lut_u[(v >> 8) & 0xF])) << 16) |
                (static_cast<uint32_t>(__ldg(&lut_u[(v >> 12) & 0xF])));
        res.z = (static_cast<uint32_t>(__ldg(&lut_u[(v >> 16) & 0xF])) << 16) |
                (static_cast<uint32_t>(__ldg(&lut_u[(v >> 20) & 0xF])));
        res.w = (static_cast<uint32_t>(__ldg(&lut_u[(v >> 24) & 0xF])) << 16) |
                (static_cast<uint32_t>(__ldg(&lut_u[(v >> 28) & 0xF])));
        return res;
    };
    uint4* out_u4 = reinterpret_cast<uint4*>(out);
    for (auto i = offset; i < n; i += stride) {
        // Each iteration, read 4 * 4 = 16 bytes, and write 32 * 2-byte outputs
        auto v = in[i];
        out_u4[4 * i + 0] = lookup_x8(v.x);
        out_u4[4 * i + 1] = lookup_x8(v.y);
        out_u4[4 * i + 2] = lookup_x8(v.z);
        out_u4[4 * i + 3] = lookup_x8(v.w);
    }
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

struct half8 {
    half2 data[4];
};
__device__ inline half8 dequant_linear4_fp16(uint q) {
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
    out.data[0] = reinterpret<half2>(v04) - a;
    out.data[1] = __hfma2(reinterpret<half2>(v15), b, c);
    out.data[2] = reinterpret<half2>(v26) - a;
    out.data[3] = __hfma2(reinterpret<half2>(v37), b, c);
    return out;
}

__global__ void kernel__copy_linear4_fp16(const uint4* in, half* out, size_t n) {
    const auto offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    uint4* out_u4 = reinterpret_cast<uint4*>(out);
    for (auto i = offset; i < n; i += stride) {
        // Each iteration, read 4 * 4 = 16 bytes, and write 32 * fp16 outputs
        auto v = in[i];
        out_u4[4 * i + 0] = reinterpret<uint4>(dequant_linear4_fp16(v.x).data);
        out_u4[4 * i + 1] = reinterpret<uint4>(dequant_linear4_fp16(v.y).data);
        out_u4[4 * i + 2] = reinterpret<uint4>(dequant_linear4_fp16(v.z).data);
        out_u4[4 * i + 3] = reinterpret<uint4>(dequant_linear4_fp16(v.w).data);
    }
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
void expect_eq(const T& a, const T& b, const std::string& msg, T tolerance) {
    if (std::abs(a - b) > tolerance) {
        std::ostringstream oss;
        oss << "Error, expected: " << a << ", actual: " << b << " (tolerance " << tolerance
            << "),  " << msg;
        throw std::runtime_error(oss.str());
    }
}

void test_lut() {
    // Problem
    thrust::host_vector<__nv_bfloat16> h_lut4(16);
    for (int i = 0; i < 16; i++) {
        h_lut4[i] = __float2bfloat16((i - 8) * 10);
    }
    thrust::device_vector<__nv_bfloat16> d_lut4 = h_lut4;
    thrust::device_vector<uint8_t> d_in({0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF});
    thrust::host_vector<float> expected({-80.0f, -70.0f, -60.0f, -50.0f, -40.0f, -30.0f, -20.0f,
                                         -10.0f, 0.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f,
                                         70.0f});
    // Double up the data - once to fill an uint4, more for threads & loops
    for (auto i = 0; i < 4; i++) {
        d_in.insert(d_in.end(), d_in.begin(), d_in.end());
        expected.insert(expected.end(), expected.begin(), expected.end());
    }
    const float sentinel = 1.96875f;
    expected.push_back(sentinel);  // sentinel at end to check no overrun

    // 4-bit LUT
    {
        thrust::device_vector<__nv_bfloat16> d_out(expected.size());
        d_out.back() = __float2bfloat16(sentinel);
        kernel__copy_lut4<<<2, 2>>>(reinterpret_cast<const uint4*>(d_in.data().get()),
                                    d_lut4.data().get(), d_out.data().get(), d_in.size() / 16);
        CHECK_CUDA(cudaGetLastError());
        thrust::host_vector<__nv_bfloat16> h_out = d_out;
        for (size_t i = 0; i < expected.size(); i++) {
            float v = __bfloat162float(h_out[i]);
            expect_eq(v, expected[i], "at index " + std::to_string(i), 1e-3f);
        }
    }

    // 8-bit LUT
    {
        thrust::host_vector<__nv_bfloat162> h_lut8(256);
        for (int i = 0; i < 256; i++) {
            h_lut8[i] = __nv_bfloat162(h_lut4[i / 16], h_lut4[i % 16]);
        }
        thrust::device_vector<__nv_bfloat162> d_lut8 = h_lut8;

        thrust::device_vector<__nv_bfloat16> d_out(expected.size());
        d_out.back() = __float2bfloat16(sentinel);
        kernel__copy_lut8<<<2, 2>>>(
            reinterpret_cast<const uint4*>(d_in.data().get()), d_lut8.data().get(),
            reinterpret_cast<__nv_bfloat162*>(d_out.data().get()), d_in.size() / 16);
        CHECK_CUDA(cudaGetLastError());
        thrust::host_vector<__nv_bfloat16> h_out = d_out;
        for (size_t i = 0; i < expected.size(); i++) {
            float v = __bfloat162float(h_out[i]);
            expect_eq(v, expected[i], "at index " + std::to_string(i), 1e-3f);
        }
    }
}

void test_linear4_fp16() {
    // Problem
    thrust::device_vector<uint8_t> d_in({0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF});
    // Order defined by the shuffle in dequant_linear4_fp16
    thrust::host_vector<float> expected({-7, -3, -8, -4, -5, -1, -6, -2, 1, 5, 0, 4, 3, 7, 2, 6});
    // Double up the data - once to fill an uint4, more for threads & loops
    for (auto i = 0; i < 4; i++) {
        d_in.insert(d_in.end(), d_in.begin(), d_in.end());
        expected.insert(expected.end(), expected.begin(), expected.end());
    }
    const float sentinel = 1.96875f;
    expected.push_back(sentinel);  // sentinel at end to check no overrun

    // Test
    thrust::device_vector<half> d_out(expected.size());
    d_out.back() = __float2half(sentinel);
    kernel__copy_linear4_fp16<<<2, 2>>>(reinterpret_cast<const uint4*>(d_in.data().get()),
                                        d_out.data().get(), d_in.size() / 16);
    CHECK_CUDA(cudaGetLastError());
    thrust::host_vector<half> h_out = d_out;
    for (size_t i = 0; i < h_out.size(); i++) {
        float v = __half2float(h_out[i]);
        expect_eq(v, expected[i], "at index " + std::to_string(i), 1e-3f);
    }
}

// --------------------------
// Benchmarking

double measure_time(uint reps, cudaStream_t stream, const std::function<void()>& fn) {
    for (uint i = 0; i < reps; i++) {
        fn();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (uint i = 0; i < reps; i++) {
        fn();
    }
    CHECK_CUDA(cudaEventRecord(end, stream));
    CHECK_CUDA(cudaEventSynchronize(end));

    float time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, end));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));

    return time_ms * 1e-3 / reps;
}

// --------------------------
// Driver program

int main() {
    // Problem size
    constexpr size_t bytes = 4 * (1ull << 30);
    static_assert(bytes % 16 == 0, "bytes must be a multiple of 16 for uint4 fill");
    const uint reps = 10;
    // std::regex pattern("^(copy|copy_lut8|copy_lut4|copy_linear4_fp16)$");
    std::regex pattern(".*");

    // Device info
    auto device_name = get_device_name();
    auto cuda_version = get_cuda_version();
    std::cerr << "Using device: " << device_name << std::endl;
    std::cerr << "CUDA version: " << cuda_version << std::endl;
    std::cerr << "Bytes: " << bytes << " (" << (bytes / (1ull << 30)) << " GiB)" << std::endl;
    std::cerr << std::endl;

    test_lut();
    test_linear4_fp16();

    // Benchmarking
    Log log;
    Stream stream;
    thrust::device_vector<uint8_t> d_a(bytes);
    thrust::device_vector<uint8_t> d_b(bytes);
    thrust::sequence(thrust::cuda::par.on(stream), d_a.begin(), d_a.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), d_b.begin(), d_b.end(), 0);

    auto run_benchmark = [&](const std::string& name, size_t bytes_read, size_t bytes_write,
                             const std::function<void()>& fn) {
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
             {"reps", static_cast<uint64_t>(reps)},
             {"id", log.id},
             {"device", device_name},
             {"cuda_version", cuda_version}});
    };

    std::cerr << "### Bench" << std::endl;

    // Baseline measurements

    run_benchmark("cudamemcpy", bytes, bytes, [&]() {
        CHECK_CUDA(cudaMemcpyAsync(d_b.data().get(), d_a.data().get(), bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    });

    run_benchmark("copy", bytes, bytes, [&]() {
        int threads = 1024;
        int blocks = 65536;
        const size_t n4 = bytes / sizeof(uint4);
        kernel__copy<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint4*>(d_a.data().get()),
            reinterpret_cast<uint4*>(d_b.data().get()), n4);
        CHECK_CUDA(cudaGetLastError());
    });

    thrust::device_vector<unsigned long long> d_sum(1);
    run_benchmark("read_reduce_sum", bytes, sizeof(unsigned long long), [&]() {
        CHECK_CUDA(cudaMemsetAsync(d_sum.data().get(), 0, sizeof(unsigned long long), stream));
        int threads = 1024;
        int blocks = 16384;
        const size_t n = bytes / sizeof(uint32_t);
        kernel__read_reduce_u32<<<blocks, threads, threads * sizeof(unsigned long long), stream>>>(
            reinterpret_cast<const uint32_t*>(d_a.data().get()), n, d_sum.data().get());
        CHECK_CUDA(cudaGetLastError());
    });

    run_benchmark("write_fill", sizeof(uint4), bytes, [&]() {
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
        const auto n = bytes / sizeof(__nv_bfloat16);
        thrust::device_vector<__nv_bfloat16> d_b16(n);
        thrust::host_vector<__nv_bfloat162> h_lut8(256);
        for (int i = 0; i < 256; i++) {
            h_lut8[i] = __nv_bfloat162{i % 16, i / 16};
        }
        thrust::device_vector<__nv_bfloat162> d_lut8(256);
        thrust::copy(h_lut8.begin(), h_lut8.end(), d_lut8.begin());
        run_benchmark("copy_lut8", n / 2, n * sizeof(__nv_bfloat16), [&]() {
            int threads = 1024;
            int blocks = 65536;
            kernel__copy_lut8<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_a.data().get()), d_lut8.data().get(),
                reinterpret_cast<__nv_bfloat162*>(d_b16.data().get()), n / 32);
            CHECK_CUDA(cudaGetLastError());
        });
    }

    // LUT4
    {
        const auto n = bytes / sizeof(__nv_bfloat16);
        thrust::device_vector<__nv_bfloat16> d_b16(n);
        thrust::host_vector<__nv_bfloat16> h_lut4(16);
        for (int i = 0; i < 16; i++) {
            h_lut4[i] = __nv_bfloat16(i);
        }
        thrust::device_vector<__nv_bfloat16> d_lut4(16);
        thrust::copy(h_lut4.begin(), h_lut4.end(), d_lut4.begin());
        run_benchmark("copy_lut4", n / 2, n * sizeof(__nv_bfloat16), [&]() {
            int threads = 1024;
            int blocks = 65536;
            kernel__copy_lut4<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_a.data().get()), d_lut4.data().get(),
                reinterpret_cast<__nv_bfloat16*>(d_b16.data().get()), n / 32);
            CHECK_CUDA(cudaGetLastError());
        });
    }

    // Linear4 fp16
    {
        const auto n = bytes / sizeof(half);
        thrust::device_vector<half> d_fp16(n);
        run_benchmark("copy_linear4_fp16", n / 2, n * sizeof(half), [&]() {
            int threads = 1024;
            int blocks = 65536;
            kernel__copy_linear4_fp16<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_a.data().get()), d_fp16.data().get(), n / 32);
            CHECK_CUDA(cudaGetLastError());
        });
    }

    return 0;
}
