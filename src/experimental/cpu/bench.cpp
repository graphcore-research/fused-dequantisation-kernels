// Arm CPU kernels and benchmarks

#if !defined(__ARM_NEON)
#error "This benchmark requires ARM NEON support."
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include <arm_neon.h>
#include <omp.h>

#define NOINLINE __attribute__((noinline))
using bf16 = __bf16;

// ----------------------------------------------------------------------------
// Kernels

template <class T>
NOINLINE void kernel_memcpy(T* __restrict__ dst, const T* __restrict__ src, uint64_t n) {
#pragma omp parallel for
    for (auto i = 0ull; i < n; ++i) {
        dst[i] = src[i];
    }
}

template <class T>
NOINLINE void kernel_reduce_sum(const T* __restrict__ src, uint64_t n, float* __restrict__ result) {
    float sum = 0.0f;
#pragma omp parallel for
    for (auto i = 0ull; i < n; ++i) {
#pragma float_control(precise, off)
        sum += float(src[i]);
    }
    *result = sum;
}

float dot(const bf16* __restrict__ a, const bf16* __restrict__ b, const uint64_t dK) {
    float32x4_t acc = vmovq_n_f32(0.0f);
    // Main loop, process 8 elements per iteration
    const auto kStop = (dK / 8) * 8;
    for (auto k = 0u; k < kStop; k += 8) {
        auto ai = vld1q_bf16(a + k);
        auto bi = vld1q_bf16(b + k);
        acc = vbfdotq_f32(acc, ai, bi);
    }
    // Handle remainder when dK is not a multiple of 8
    float result = vaddvq_f32(acc);
    for (auto k = kStop; k < dK; ++k) {
        result += float(a[k]) * float(b[k]);
    }
    return result;
}

NOINLINE void kernel_mv(const bf16* __restrict__ a,  // [dK]
                        const bf16* __restrict__ b,  // [dN * dK]
                        const uint64_t dK,
                        const uint64_t dN,
                        bf16* __restrict__ out) {  // [dN]
#pragma omp parallel for
    for (auto n = 0ull; n < dN; ++n) {
        out[n] = bf16(dot(a, &b[n * dK], dK));
    }
    return;
}

template <uint BlockM, uint BlockN>
void _mm_chunk_bfmmla(const bf16* __restrict__ a,
                      const bf16* __restrict__ b,
                      const uint64_t dK,
                      const uint64_t dN,
                      bf16* __restrict__ out) {
    // Note: we expect all BlockM, BlockN loops to be unrolled
    static_assert(BlockM % 2 == 0 && BlockN % 2 == 0, "BlockM and BlockN must be even");

    // Each accumulator holds a 2x2 result, accumulated over the full `k` dimension
    float32x4_t accs[(BlockM / 2) * (BlockN / 2)];
    for (auto i = 0u; i < (BlockM / 2) * (BlockN / 2); ++i) {
        accs[i] = vmovq_n_f32(0.0f);
    }
    // Main loop, process `(m, k, n) = (BlockM, 8, BlockN)` elements per iteration
    const auto kStop = (dK / 8) * 8;
    for (auto k = 0u; k < kStop; k += 8) {
        bfloat16x8_t aa[BlockM], bb[BlockN];
        for (auto m = 0u; m < BlockM; ++m) {
            aa[m] = vld1q_bf16(&a[m * dK + k]);
        }
        for (auto n = 0u; n < BlockN; ++n) {
            bb[n] = vld1q_bf16(&b[n * dK + k]);
        }
        for (auto m = 0u; m < (BlockM / 2); ++m) {
            for (auto n = 0u; n < (BlockN / 2); ++n) {
                auto& acc = accs[m * (BlockN / 2) + n];
                acc = vbfmmlaq_f32(
                    acc, vcombine_bf16(vget_low_bf16(aa[2 * m]), vget_low_bf16(aa[2 * m + 1])),
                    vcombine_bf16(vget_low_bf16(bb[2 * n]), vget_low_bf16(bb[2 * n + 1])));
                acc = vbfmmlaq_f32(
                    acc, vcombine_bf16(vget_high_bf16(aa[2 * m]), vget_high_bf16(aa[2 * m + 1])),
                    vcombine_bf16(vget_high_bf16(bb[2 * n]), vget_high_bf16(bb[2 * n + 1])));
            }
        }
    }
    // Handle remainder when dK is not a multiple of 8
    for (auto k = kStop; k < dK; ++k) {
        for (auto m = 0u; m < (BlockM / 2); ++m) {
            for (auto n = 0u; n < (BlockN / 2); ++n) {
                auto& acc = accs[m * (BlockN / 2) + n];
                float a0 = float(a[(2 * m + 0) * dK + k]);
                float a1 = float(a[(2 * m + 1) * dK + k]);
                float b0 = float(b[(2 * n + 0) * dK + k]);
                float b1 = float(b[(2 * n + 1) * dK + k]);
                acc = vmlaq_f32(acc, float32x4_t{a0, a0, a1, a1}, float32x4_t{b0, b1, b0, b1});
            }
        }
    }
    // Store out results, a BlockM x BlockN matrix
    for (auto m = 0u; m < (BlockM / 2); ++m) {
        for (auto n = 0u; n < (BlockN / 2); ++n) {
            auto acc_bf16 = vcvt_bf16_f32(accs[m * (BlockN / 2) + n]);
            vst1_lane_bf16(&out[(2 * m + 0) * dN + (2 * n + 0)], acc_bf16, 0);
            vst1_lane_bf16(&out[(2 * m + 0) * dN + (2 * n + 1)], acc_bf16, 1);
            vst1_lane_bf16(&out[(2 * m + 1) * dN + (2 * n + 0)], acc_bf16, 2);
            vst1_lane_bf16(&out[(2 * m + 1) * dN + (2 * n + 1)], acc_bf16, 3);
        }
    }
}

void kernel_mm(const bf16* __restrict__ a,
               const bf16* __restrict__ b,
               const uint64_t dM,
               const uint64_t dK,
               const uint64_t dN,
               bf16* __restrict__ out) {
    // Matrix-vector case
    if (dM == 1 || dN == 1) {
#pragma omp parallel for
        for (auto i = 0ull; i < dM * dN; ++i) {
            auto m = i / dN;
            auto n = i % dN;
            out[m * dN + n] = bf16(dot(&a[m * dK], &b[n * dK], dK));
        }
        return;
    }

    constexpr auto G0 = 16ull;  // block size
    constexpr auto G1 = 8ull;   // inner block size

    const auto blocksM = (dM + G0 - 1) / G0;
    const auto blocksN = (dN + G0 - 1) / G0;

#pragma omp parallel for
    for (auto i0 = 0ull; i0 < blocksM * blocksN; ++i0) {
        auto m0 = G0 * (i0 / blocksN);
        auto m1 = std::min<uint64_t>(m0 + G0, dM);
        auto n0 = G0 * (i0 % blocksN);
        auto n1 = std::min<uint64_t>(n0 + G0, dN);

        // Main loop
        auto mStop = (m1 / G1) * G1;
        auto nStop = (n1 / G1) * G1;
        for (auto n = n0; n < nStop; n += G1) {
            for (auto m = m0; m < mStop; m += G1) {
                _mm_chunk_bfmmla<G1, G1>(&a[m * dK], &b[n * dK], dK, dN, &out[m * dN + n]);
            }
        }
        // Handle remainder when dN is not a multiple of G1, `out[m0:m1, nStop:n1]`
        for (auto n = nStop; n < n1; ++n) {
            for (auto m = m0; m < m1; ++m) {
                out[m * dN + n] = bf16(dot(&a[m * dK], &b[n * dK], dK));
            }
        }
        // Handle remainder when dM is not a multiple of G1, `out[mStop:m1, n0:nStop]`
        // (note: excludes the bottom-right corner which is handled in the loop above)
        for (auto m = mStop; m < m1; ++m) {
            for (auto n = n0; n < nStop; ++n) {
                out[m * dN + n] = bf16(dot(&a[m * dK], &b[n * dK], dK));
            }
        }
    }
}

void kernel_mm_naive(const bf16* __restrict__ a,  // [dM * dK]
                     const bf16* __restrict__ b,  // [dN * dK]
                     const uint64_t dM,
                     const uint64_t dK,
                     const uint64_t dN,
                     bf16* __restrict__ out) {  // [dM * dN]
    for (auto m = 0u; m < dM; ++m) {
        for (auto n = 0u; n < dN; ++n) {
            auto sum = 0.0f;
            for (auto k = 0u; k < dK; ++k) {
                sum += float(a[m * dK + k]) * float(b[n * dK + k]);
            }
            out[m * dN + n] = bf16(sum);
        }
    }
}

// ----------------------------------------------------------------------------
// Tests

template <class T>
std::vector<T> arange(uint64_t n) {
    std::vector<T> v(n);
    for (uint64_t i = 0; i < n; ++i) {
        v[i] = T(i);
    }
    return v;
}

template <class T>
std::vector<T> randn(uint64_t n, std::default_random_engine& rng) {
    std::normal_distribution<float> dist(0.0, 1.0);
    std::vector<T> v(n);
    for (uint64_t i = 0; i < n; ++i) {
        v[i] = T(dist(rng));
    }
    return v;
}

#define EXPECT_EQ(expected, actual, tol) expect_eq(expected, actual, tol, __FILE__, __LINE__)

void expect_eq(const std::vector<bf16>& expected,
               const std::vector<bf16>& actual,
               float tol,
               const char* file,
               int line) {
    if (expected.size() != actual.size()) {
        std::ostringstream str;
        str << "EXPECT_EQ size mismatch: " << expected.size() << " != " << actual.size() << " at "
            << file << ":" << line << std::endl;
        throw std::runtime_error(str.str());
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(float(expected[i]) - float(actual[i])) > tol * std::abs(float(expected[i]))) {
            std::ostringstream str;
            str << "EXPECT_EQ value mismatch at index " << i << ": " << float(expected[i])
                << " != " << float(actual[i]) << " at " << file << ":" << line << std::endl;
            throw std::runtime_error(str.str());
        }
    }
}

void test_kernel_mm() {
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> sizes = {
        // dM, dK, dN
        {1, 100, 200},  //
        {100, 200, 1},  //
        {32, 16, 64},   //
        {15, 133, 63},  //
        {10, 20, 30},   //
    };
    for (auto [dM, dK, dN] : sizes) {
        std::default_random_engine rng(100);
        auto a = randn<bf16>(dM * dK, rng);
        auto b = randn<bf16>(dN * dK, rng);
        std::vector<bf16> out(dM * dN);
        kernel_mm(a.data(), b.data(), dM, dK, dN, out.data());

        std::vector<bf16> expected(dM * dN);
        kernel_mm_naive(a.data(), b.data(), dM, dK, dN, expected.data());
        EXPECT_EQ(expected, out, 1e-3f);

        if (dM == 1) {
            // Also test the MV kernel
            std::vector<bf16> out_mv(dN);
            kernel_mv(a.data(), b.data(), dK, dN, out_mv.data());
            EXPECT_EQ(expected, out_mv, 1e-3f);
        }
    }
}

// ----------------------------------------------------------------------------
// Benchmarks

struct TimingStats {
    double avg_time;
    double avg_time_stderr;
};

TimingStats measure_time(uint64_t reps, const std::function<void(uint64_t)>& fn) {
    // Warmup
    for (auto i = 0ull; i < reps; i++) {
        fn(i);
    }

    // Timing
    std::vector<double> samples;
    for (auto i = 0ull; i < reps; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        fn(i);
        auto end = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double>(end - start).count());
    }

    auto mean = std::accumulate(samples.begin(), samples.end(), 0.0) / double(samples.size());
    auto variance = 0.0;
    for (auto sample : samples) {
        variance += std::pow(sample - mean, 2);
    }
    variance /= static_cast<double>(samples.size() - 1);
    auto stderr_ = std::sqrt(variance / double(samples.size()));
    return {mean, stderr_};
}

void benchmark_memcpy() {
    std::cerr << "### benchmark_memcpy" << std::endl;

    const uint64_t bytes_to_copy = 256 * 1024 * 1024;
    const uint64_t copies = (1ull << 30) / bytes_to_copy;  // to avoid caching effects
    const uint64_t reps = 16;

    // Allocate
    const uint64_t n_elems = bytes_to_copy / sizeof(bf16);
    std::vector<bf16> src(copies * n_elems, bf16(0.5f));
    std::vector<bf16> dst(copies * n_elems);

    // Benchmark
    auto s = measure_time(reps, [&](uint64_t i) {
        auto idx = i % copies;
        kernel_memcpy(&dst[idx * n_elems], &src[idx * n_elems], n_elems);
    });
    double gbs = 2 * double(bytes_to_copy) / (s.avg_time * 1e9);
    std::cerr << std::left << std::setw(20)
              << (std::to_string(bytes_to_copy / (1024 * 1024)) + " MB") << std::right
              << std::setw(15) << std::fixed << std::setprecision(3) << (s.avg_time * 1e3)
              << " ms"  //
              << std::right << std::setw(15) << std::fixed << std::setprecision(1) << gbs
              << " GB/s"  //
              << "\n"
              << std::endl;
}

void benchmark_reduce_sum() {
    std::cerr << "### benchmark_reduce_sum" << std::endl;

    const uint64_t n_elems = 16 * 1024 * 1024;
    const uint64_t copies = (1ull << 30) / (n_elems * sizeof(bf16));
    const uint64_t reps = 16;

    // Allocate
    std::vector<bf16> src(copies * n_elems, bf16(0.5f));

    // Benchmark
    float result;
    auto s = measure_time(reps, [&](uint64_t i) {
        auto idx = i % copies;
        kernel_reduce_sum(&src[idx * n_elems], n_elems, &result);
    });
    double gbs = double(n_elems * sizeof(bf16)) / (s.avg_time * 1e9);
    std::cerr << std::left << std::setw(20) << std::to_string(n_elems) + " elements"  //
              << std::right << std::setw(15) << std::fixed << std::setprecision(3)
              << (s.avg_time * 1e3) << " ms"  //
              << std::right << std::setw(15) << std::fixed << std::setprecision(1) << gbs
              << " GB/s"  //
              << "\n"
              << std::endl;
}

void benchmark_mv() {
    std::cerr << "### benchmark_mv" << std::endl;

    const std::vector<std::tuple<uint64_t, uint64_t>> sizes = {
        {4096, 4096},
        {8192, 8192},
    };
    const uint64_t reps = 16;

    for (const auto& [dK, dN] : sizes) {
        // Allocate
        auto copies = (1ull << 30) / (dK * dN * sizeof(bf16));
        std::vector<bf16> a(copies * dK, bf16(0.5f));
        std::vector<bf16> b(copies * dN * dK, bf16(0.5f));
        std::vector<bf16> out(copies * dN);

        // Benchmark
        auto s = measure_time(reps, [&](uint64_t i) {
            auto idx = i % copies;
            kernel_mv(&a[idx * dK], &b[idx * dN * dK], dK, dN, &out[idx * dN]);
        });
        double bytes = double(dK + dK * dN + dN) * sizeof(bf16);
        double gbs = bytes / (s.avg_time * 1e9);
        std::cerr << std::left << std::setw(20)
                  << (std::to_string(dK) + "x" + std::to_string(dN))  //
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3)
                  << (s.avg_time * 1e3) << " ms"  //
                  << std::right << std::setw(15) << std::fixed << std::setprecision(1) << gbs
                  << " GB/s"  //
                  << std::endl;
    }
    std::cerr << std::endl;
}

void benchmark_mm() {
    std::cerr << "### benchmark_mm" << std::endl;

    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> sizes = {
        // dM, dK, dN
        {1, 4096, 4096},
        {16, 4096, 4096},
        {256, 4096, 4096},
        {1024, 4096, 4096},
    };
    const uint64_t reps = 16;

    for (const auto& [dM, dK, dN] : sizes) {
        // Allocate
        auto copies = (1ull << 30) / (dK * dN * sizeof(bf16));  // assume dM << dK, dN
        std::vector<bf16> a(copies * dM * dK, bf16(0.5f));
        std::vector<bf16> b(copies * dN * dK, bf16(0.5f));
        std::vector<bf16> out(copies * dM * dN);

        // Benchmark
        auto s = measure_time(reps, [&](uint64_t i) {
            auto idx = i % copies;
            kernel_mm(&a[idx * dM * dK], &b[idx * dN * dK], dM, dK, dN, &out[idx * dM * dN]);
        });
        double gbs = double(dM * dK + dN * dK + dM * dN) * sizeof(bf16) / (s.avg_time * 1e9);
        double flops = 2.0 * double(dM * dN * dK) / (s.avg_time * 1e9);
        std::cerr << std::left << std::setw(20)
                  << (std::to_string(dM) + "x" + std::to_string(dK) + "x" + std::to_string(dN))  //
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3)
                  << (s.avg_time * 1e3) << " ms"  //
                  << std::right << std::setw(15) << std::fixed << std::setprecision(1) << gbs
                  << " GB/s"  //
                  << std::right << std::setw(15) << std::fixed << std::setprecision(1) << flops
                  << " GFLOP/s"  //
                  << std::endl;
    }
    std::cerr << std::endl;
}

// ----------------------------------------------------------------------------
// Driver program

void run_tests() {
    std::cerr << "### Running tests" << std::endl;
    test_kernel_mm();
    std::cerr << std::endl;
}

int main() {
    // auto threads = 1;
    auto threads = std::thread::hardware_concurrency();
    omp_set_num_threads(int(threads));
    std::cerr << "# Using " << threads << " threads" << std::endl << std::endl;

    run_tests();

    benchmark_memcpy();
    benchmark_reduce_sum();
    benchmark_mv();
    benchmark_mm();

    return 0;
}
