// Arm CPU kernels and benchmarks

#if !defined(__ARM_NEON)
#error "This benchmark requires ARM NEON support."
#endif

#include <algorithm>
#include <cassert>
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

std::ostream& operator<<(std::ostream& out, const bf16& v) {
    return out << float(v);
}

template <class T, size_t N>
std::ostream& operator<<(std::ostream& out, const std::array<T, N>& arr) {
    out << "[";
    for (size_t i = 0; i < N; ++i) {
        if (i) out << ", ";
        out << arr[i];
    }
    out << "]";
    return out;
}

template <class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    out << "[";
    for (auto i = 0ull; i < v.size(); ++i) {
        if (i) out << ", ";
        out << v[i];
    }
    out << "]";
    return out;
}

// ----------------------------------------------------------------------------
// Kernels

namespace kernels {

template <class T>
NOINLINE void memcpy(T* __restrict__ dst, const T* __restrict__ src, uint64_t n) {
#pragma omp parallel for
    for (auto i = 0ull; i < n; ++i) {
        dst[i] = src[i];
    }
}

template <class T>
NOINLINE void reduce_sum(const T* __restrict__ src, uint64_t n, float* __restrict__ result) {
    float sum = 0.0f;
#pragma omp parallel for
    for (auto i = 0ull; i < n; ++i) {
#pragma float_control(precise, off)
        sum += float(src[i]);
    }
    *result = sum;
}

float _dot(const bf16* __restrict__ a, const bf16* __restrict__ b, const uint64_t dK) {
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

NOINLINE void mv(const bf16* __restrict__ a,  // [dK]
                 const bf16* __restrict__ b,  // [dN * dK]
                 const uint64_t dK,
                 const uint64_t dN,
                 bf16* __restrict__ out) {  // [dN]
#pragma omp parallel for
    for (auto n = 0ull; n < dN; ++n) {
        out[n] = bf16(_dot(a, &b[n * dK], dK));
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

void mm(const bf16* __restrict__ a,
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
            out[m * dN + n] = bf16(_dot(&a[m * dK], &b[n * dK], dK));
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
                out[m * dN + n] = bf16(_dot(&a[m * dK], &b[n * dK], dK));
            }
        }
        // Handle remainder when dM is not a multiple of G1, `out[mStop:m1, n0:nStop]`
        // (note: excludes the bottom-right corner which is handled in the loop above)
        for (auto m = mStop; m < m1; ++m) {
            for (auto n = n0; n < nStop; ++n) {
                out[m * dN + n] = bf16(_dot(&a[m * dK], &b[n * dK], dK));
            }
        }
    }
}

void mm_naive(const bf16* __restrict__ a,  // [dM * dK]
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

// LUT8

float _dot_lut8(const bf16* __restrict__ a,
                const uint32_t* __restrict__ b,
                const uint32_t* __restrict__ lut8,
                const bf16* __restrict__ bs,
                const uint64_t dK,
                const uint64_t dG) {
    float32x4_t accu = vmovq_n_f32(0.0f);
    float32x4_t acc = vmovq_n_f32(0.0f);
    for (auto g = 0u; g < (dK / dG); ++g) {
        // Main loop, process 8 elements per iteration
        for (auto k = dG * g; k < dG * (g + 1); k += 8) {
            auto ai = vld1q_bf16(a + k);
            auto biq = b[k / 8];
            bf16 bia[8];
            reinterpret_cast<uint32_t&>(bia[0]) = lut8[(biq >> 0) & 0xFF];
            reinterpret_cast<uint32_t&>(bia[2]) = lut8[(biq >> 8) & 0xFF];
            reinterpret_cast<uint32_t&>(bia[4]) = lut8[(biq >> 16) & 0xFF];
            reinterpret_cast<uint32_t&>(bia[6]) = lut8[(biq >> 24) & 0xFF];
            auto biu = vld1q_bf16(&bia[0]);
            accu = vbfdotq_f32(accu, ai, biu);
        }
        auto bsi = bs[g];
        acc = vmlaq_f32(acc, accu, vdupq_n_f32(float(bsi)));
        accu = vmovq_n_f32(0.0f);
    }
    return vaddvq_f32(acc);
}

NOINLINE void mv_lut8(const bf16* __restrict__ a,         // [dK]
                      const uint32_t* __restrict__ b,     // [dN * (dK/8)]
                      const uint32_t* __restrict__ lut8,  // [256]
                      const bf16* __restrict__ bs,        // [dN * (dK/dG)]
                      const uint64_t dK,
                      const uint64_t dN,
                      const uint64_t dG,
                      bf16* __restrict__ out) {  // [dN]
    assert(dK % dG == 0);
    assert(dG % 8 == 0);
#pragma omp parallel for
    for (auto n = 0ull; n < dN; ++n) {
        out[n] = bf16(_dot_lut8(a, &b[n * (dK / 8)], lut8, &bs[n * (dK / dG)], dK, dG));
    }
    return;
}

}  // namespace kernels

// ----------------------------------------------------------------------------
// Tests

namespace tests {

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

double rmse_norm(const std::vector<bf16>& expected, const std::vector<bf16>& actual) {
    assert(expected.size() == actual.size());
    double diff_sq = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < expected.size(); i++) {
        auto ve = double(expected[i]);
        auto va = double(actual[i]);
        diff_sq += std::pow(ve - va, 2);
        sum_sq += ve * ve;
    }
    return std::sqrt(diff_sq / sum_sq);
}

#define EXPECT_CLOSE(expected, actual, tol) expect_close(expected, actual, tol, __FILE__, __LINE__)

void expect_close(const std::vector<bf16>& expected,
                  const std::vector<bf16>& actual,
                  double tol,
                  const char* file,
                  int line) {
    auto error = rmse_norm(expected, actual);
    if (error > tol) {
        std::ostringstream str;
        str << "EXPECT_CLOSE failed: RMSE norm " << error << " > " << tol << " at " << file << ":"
            << line << std::endl;
        throw std::runtime_error(str.str());
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
        kernels::mm(a.data(), b.data(), dM, dK, dN, out.data());

        std::vector<bf16> expected(dM * dN);
        kernels::mm_naive(a.data(), b.data(), dM, dK, dN, expected.data());
        EXPECT_EQ(expected, out, 1e-3f);

        if (dM == 1) {
            // Also test the MV kernel
            std::vector<bf16> out_mv(dN);
            kernels::mv(a.data(), b.data(), dK, dN, out_mv.data());
            EXPECT_EQ(expected, out_mv, 1e-3f);
        }
    }
}

struct ScaledTensor {
    std::vector<uint8_t> data;
    std::vector<bf16> scale;
    std::vector<bf16> lut;
    size_t block_size;
    size_t bits_per_element;

    // A N*B-bit LUT, mapping indices [0, 2^(N*B)-1) -> N x bf16
    template <uint64_t N>
    std::vector<std::array<bf16, N>> lutN() const {
        static_assert(N >= 2, "N must be at least 2");
        const auto entries = 1ull << (N * bits_per_element);
        const auto lut_mask = (1ull << bits_per_element) - 1;
        std::vector<std::array<bf16, N>> out(entries);
        for (auto i = 0ull; i < out.size(); i++) {
            for (auto j = 0ull; j < N; j++) {
                out[i][j] = bf16(lut[(i >> (j * bits_per_element)) & lut_mask]);
            }
        }
        return out;
    }

    static ScaledTensor quantise_linear(const std::vector<bf16>& input,
                                        uint64_t block_size,
                                        uint64_t bits_per_element) {
        assert(input.size() % block_size == 0);
        assert(block_size % 2 == 0);
        assert(bits_per_element < 8);
        assert(8 % bits_per_element == 0);

        const auto elements_per_byte = 8 / bits_per_element;
        const int v_min = -(1 << (bits_per_element - 1));
        const int v_max = -v_min - (bits_per_element > 1);

        const auto n = input.size();
        std::vector<uint8_t> data(n / elements_per_byte);
        std::vector<bf16> scale(n / block_size);
        for (auto g = 0ull; g < n / block_size; g++) {
            // Find scale
            float max_abs = 1e-12f, sum_abs = 1e-12f;
            for (auto i = 0ull; i < block_size; i++) {
                float vi = std::abs(float(input[g * block_size + i]));
                max_abs = std::max(max_abs, vi);
                sum_abs += vi;
            }
            scale[g] = bf16((bits_per_element == 1) ? (sum_abs / float(block_size))
                                                    : (max_abs / float(v_max)));

            // Quantise
            for (auto i = 0ull; i < block_size; i++) {
                float v = float(input[g * block_size + i]) / float(scale[g]);
                uint8_t q;
                if (bits_per_element == 1) {
                    q = (v >= 0);
                } else {
                    q = static_cast<uint8_t>(
                        std::clamp(static_cast<int>(std::round(v)), v_min, v_max) - v_min);
                }
                auto byte_index = (g * block_size + i) / elements_per_byte;
                auto bit_offset = (i % elements_per_byte) * bits_per_element;
                data[byte_index] |= (q << bit_offset);
            }
        }

        // Build LUT
        std::vector<bf16> lut(1 << bits_per_element);
        if (bits_per_element == 1) {
            lut = {-1, 1};
        } else {
            // Identity mapping
            for (int v = v_min; v <= v_max; v++) {
                lut[uint64_t(v - v_min)] = bf16(v);
            }
        }

        return {.data = data,
                .scale = scale,
                .lut = lut,
                .block_size = block_size,
                .bits_per_element = bits_per_element};
    }

    std::vector<bf16> dequantise() {
        const uint64_t elements_per_byte = 8 / bits_per_element;
        const auto mask = uint8_t((1 << bits_per_element) - 1);
        auto n = data.size() * elements_per_byte;
        std::vector<bf16> output(n);
        for (auto g = 0ull; g < n / block_size; g++) {
            auto s = scale[g];
            for (auto i = 0u; i < block_size; i++) {
                auto byte_index = (g * block_size + i) / elements_per_byte;
                auto bit_offset = (i % elements_per_byte) * bits_per_element;
                auto q = (data[byte_index] >> bit_offset) & mask;
                auto v = lut[uint64_t(q)] * s;
                output[g * block_size + i] = v;
            }
        }
        return output;
    }
};

void test_block_quantise() {
    auto dK = 64ull, dN = 8ull;
    auto block_size = 16ull;
    auto bits_per_element = 4ull;

    std::default_random_engine rng(200);
    auto a = randn<bf16>(dK, rng);
    auto b = randn<bf16>(dN * dK, rng);
    std::vector<bf16> original(dN), ref(dN), actual(dN);

    kernels::mv(a.data(), b.data(), dK, dN, original.data());

    auto bq = ScaledTensor::quantise_linear(b, block_size, bits_per_element);
    auto br = bq.dequantise();
    kernels::mv(a.data(), br.data(), dK, dN, ref.data());

    auto lut8 = bq.lutN<2>();
    kernels::mv_lut8(a.data(), reinterpret_cast<const uint32_t*>(bq.data.data()),
                     reinterpret_cast<const uint32_t*>(lut8.data()), bq.scale.data(), dK, dN,
                     block_size, actual.data());

    EXPECT_CLOSE(ref, actual, 0.001);
    EXPECT_CLOSE(original, actual, 0.2);  // depends on bits_per_element
}

void test_all() {
    std::cerr << "### Running tests\n\n";
    test_kernel_mm();
    test_block_quantise();
    std::cerr << "---> Tests passed\n\n" << std::flush;
}

}  // namespace tests

// ----------------------------------------------------------------------------
// Benchmarks

namespace benchmarks {

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
        kernels::memcpy(&dst[idx * n_elems], &src[idx * n_elems], n_elems);
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
        kernels::reduce_sum(&src[idx * n_elems], n_elems, &result);
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
            kernels::mv(&a[idx * dK], &b[idx * dN * dK], dK, dN, &out[idx * dN]);
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

void benchmark_mv_lut8() {
    std::cerr << "### benchmark_mv_lut8" << std::endl;

    const std::vector<std::tuple<uint64_t, uint64_t>> sizes = {
        {4096, 4096},
    };
    const uint64_t reps = 16;

    for (const auto& [dK, dN] : sizes) {
        // Allocate
        auto copies = (1ull << 30) / (dK * dN * sizeof(bf16));
        auto dG = 64ull;
        std::vector<bf16> a(copies * dK, bf16(0.5f));
        std::vector<uint32_t> b(copies * dN * (dK / 8), 0xfedc0123);
        std::vector<bf16> bs(copies * dN * (dK / dG), bf16(0.25f));
        std::vector<bf16> out(copies * dN);
        std::vector<uint32_t> lut8(256, 0xabcd1234);

        // Benchmark
        auto s = measure_time(reps, [&](uint64_t i) {
            auto idx = i % copies;
            kernels::mv_lut8(&a[idx * dK], &b[idx * dN * (dK / 8)], &lut8[0],
                             &bs[idx * dN * (dK / dG)], dK, dN, dG, &out[idx * dN]);
        });
        double bytes =
            double(dK * sizeof(bf16) + (dK / 8) * dN * sizeof(uint32_t) + dN * sizeof(bf16));
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
            kernels::mm(&a[idx * dM * dK], &b[idx * dN * dK], dM, dK, dN, &out[idx * dM * dN]);
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

}  // namespace benchmarks

// ----------------------------------------------------------------------------
// Driver program

namespace demo {

void demo_vtbl2_u8() {
    // vtbl2_u8 performs table lookup using a 16-byte table stored in two uint8x8_t registers
    // Create a 16-byte lookup table: [0, 1, 2, ..., 15]
    uint8_t table_data[16];
    for (int i = 0; i < 16; i++) {
        table_data[i] = uint8_t(i * 10);  // e.g., [0, 10, 20, ..., 150]
    }
    uint8x8x2_t table;
    table.val[0] = vld1_u8(&table_data[0]);  // first 8 bytes
    table.val[1] = vld1_u8(&table_data[8]);  // next 8 bytes

    // Create indices to lookup: [3, 7, 0, 15, 2, 8, 1, 12]
    uint8_t indices_data[8] = {3, 7, 0, 15, 2, 8, 1, 12};
    uint8x8_t indices = vld1_u8(indices_data);

    // Perform table lookup
    uint8x8_t result = vtbl2_u8(table, indices);

    // Print results
    uint8_t result_data[8];
    vst1_u8(result_data, result);
    std::cerr << "vtbl2_u8 demo: indices -> values" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cerr << "  " << int(indices_data[i]) << " -> " << int(result_data[i]) << std::endl;
    }
}

void demo_lut4_to_bf16() {
    // Map 8 4-bit indices (packed in 32 bits) through 16-entry LUT to bfloat16x8_t

    // 16-entry lookup table of bf16 values (32 bytes total)
    bf16 lut_data[16];
    for (int i = 0; i < 16; i++) {
        lut_data[i] = bf16(float(i) - 8.0f);
    }

    // Load LUT as uint8 for vtbl4_u8 (covers 32 bytes)
    const uint8_t* lut_u8 = reinterpret_cast<const uint8_t*>(lut_data);
    uint8x8x4_t lut_table;
    lut_table.val[0] = vld1_u8(&lut_u8[0]);
    lut_table.val[1] = vld1_u8(&lut_u8[8]);
    lut_table.val[2] = vld1_u8(&lut_u8[16]);
    lut_table.val[3] = vld1_u8(&lut_u8[24]);

    uint32_t packed_indices = 0x0123fedc;  // 8 4-bit indices

    // Build index vectors (each bf16 is 2 bytes, so multiply by 2)
    uint8_t indices_lo[8], indices_hi[8];
    for (int i = 0; i < 8; i++) {
        uint8_t idx = (packed_indices >> (4 * i)) & 0xF;
        indices_lo[i] = idx * 2;      // low byte
        indices_hi[i] = idx * 2 + 1;  // high byte
    }

    // Lookup both bytes and interleave
    uint8x8_t result_lo = vtbl4_u8(lut_table, vld1_u8(indices_lo));
    uint8x8_t result_hi = vtbl4_u8(lut_table, vld1_u8(indices_hi));
    uint8x8x2_t interleaved = vzip_u8(result_lo, result_hi);
    bfloat16x8_t result =
        vreinterpretq_bf16_u8(vcombine_u8(interleaved.val[0], interleaved.val[1]));

    // Print results
    bf16 result_data[8];
    vst1q_bf16(result_data, result);
    std::cerr << "LUT4 to bf16 demo: indices -> values" << std::endl;
    for (int i = 0; i < 8; i++) {
        uint8_t idx = (packed_indices >> (4 * i)) & 0xF;
        std::cerr << "  " << int(idx) << " -> " << float(result_data[i]) << std::endl;
    }
}

}  // namespace demo

int main() {
    // auto threads = 1;
    auto threads = std::thread::hardware_concurrency();
    omp_set_num_threads(int(threads));
    std::cerr << "# Using " << threads << " threads" << std::endl << std::endl;

    // demo::demo_vtbl2_u8();
    // demo::demo_lut4_to_bf16();

    tests::test_all();

    benchmarks::benchmark_memcpy();
    benchmarks::benchmark_reduce_sum();
    benchmarks::benchmark_mv();
    benchmarks::benchmark_mv_lut8();
    benchmarks::benchmark_mm();

    return 0;
}
