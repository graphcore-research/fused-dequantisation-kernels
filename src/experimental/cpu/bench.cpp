// Arm CPU kernels and benchmarks

#if !defined(__ARM_NEON)
#error "This benchmark requires ARM NEON support."
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <format>
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

bf16 to_bf16(float v) {
    return vcvth_bf16_f32(v);
}
float to_float(bf16 v) {
    return vcvtah_f32_bf16(v);
}

std::ostream& operator<<(std::ostream& out, bf16 v) {
    return out << to_float(v);
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

std::ostream& operator<<(std::ostream& out, uint8x8_t v) {
    out << "[";
    out << int(vget_lane_u8(v, 0)) << ", " << int(vget_lane_u8(v, 1)) << ", "
        << int(vget_lane_u8(v, 2)) << ", " << int(vget_lane_u8(v, 3)) << ", "
        << int(vget_lane_u8(v, 4)) << ", " << int(vget_lane_u8(v, 5)) << ", "
        << int(vget_lane_u8(v, 6)) << ", " << int(vget_lane_u8(v, 7));
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
#pragma omp parallel for reduction(+ : sum)
    for (auto i = 0ull; i < n; ++i) {
#pragma float_control(precise, off)
        sum += float(src[i]);
    }
    *result = sum;
}

float _dot(const bf16* __restrict__ a, const bf16* __restrict__ b, const uint64_t dK) {
    constexpr auto P = 8;
    float32x4_t acc[P];
    // Initialize accumulators
    for (auto p = 0; p < P; ++p) {
        acc[p] = vmovq_n_f32(0.0f);
    }
    // Main loop, process 8*P elements per iteration
    constexpr auto Stride = 8 * P;
    const auto kStop = (dK / Stride) * Stride;
    for (auto k = 0u; k < kStop; k += Stride) {
        for (auto p = 0; p < P; ++p) {
            auto ai = vld1q_bf16(a + k + p * 8);
            auto bi = vld1q_bf16(b + k + p * 8);
            acc[p] = vbfdotq_f32(acc[p], ai, bi);
        }
    }
    // Combine accumulators
    float result = 0.0f;
    for (auto p = 0; p < P; ++p) {
        result += vaddvq_f32(acc[p]);
    }
    // Handle remainder when dK is not a multiple of 8*P
    for (auto k = kStop; k < dK; ++k) {
        result += to_float(a[k]) * to_float(b[k]);
    }
    return result;
}

// Matrix-vector product between `a` [dK] vector and `b` [BlockN x dK] matrix
// => `out` [BlockN] vector
template <uint64_t BlockN, uint64_t BlockK8>
void _mv_chunk(const bf16* __restrict__ a,
               const bf16* __restrict__ b,
               const uint64_t dK,
               bf16* __restrict__ out) {
    // Initialize accumulators
    float32x4_t accs[BlockN * BlockK8];
#pragma unroll
    for (auto i = 0u; i < BlockN * BlockK8; ++i) {
        accs[i] = vmovq_n_f32(0.0f);
    }
    // Main loop, process [BlockN, BlockK8 * 8] elements of `b` per iteration
    constexpr auto StrideK = BlockK8 * 8;
    const auto kStop = (dK / StrideK) * StrideK;
    for (auto k0 = 0u; k0 < kStop; k0 += StrideK) {
#pragma unroll
        for (auto iK = 0u; iK < BlockK8; ++iK) {
            auto k = k0 + iK * 8;
            auto ai = vld1q_bf16(a + k);
#pragma unroll
            for (auto n = 0u; n < BlockN; ++n) {
                auto bi = vld1q_bf16(b + n * dK + k);
                auto& acc = accs[n * BlockK8 + iK];
                acc = vbfdotq_f32(acc, ai, bi);
            }
        }
    }
    // Accumulate partials and store results
#pragma unroll
    for (auto n = 0u; n < BlockN; ++n) {
        // Sum across BlockK8 accumulators
        auto& acc_n = accs[n * BlockK8];
#pragma unroll
        for (auto iK = 1u; iK < BlockK8; ++iK) {
            acc_n = vaddq_f32(acc_n, accs[n * BlockK8 + iK]);
        }
        auto sum = vaddvq_f32(acc_n);

        // Handle remainder when dK is not a multiple of StrideK
        for (auto k = kStop; k < dK; ++k) {
            sum += to_float(a[k]) * to_float(b[n * dK + k]);
        }
        out[n] = to_bf16(sum);
    }
}

NOINLINE void mv(const bf16* __restrict__ a,  // [dK]
                 const bf16* __restrict__ b,  // [dN * dK]
                 const uint64_t dK,
                 const uint64_t dN,
                 bf16* __restrict__ out) {  // [dN]
    constexpr auto BlockN = 8ull;
    constexpr auto BlockK8 = 2ull;
    const auto nStop = BlockN * (dN / BlockN);
#pragma omp parallel for
    for (auto n = 0ull; n < nStop; n += BlockN) {
        _mv_chunk<BlockN, BlockK8>(a, &b[n * dK], dK, &out[n]);
    }
    // Handle remainder when dN is not a multiple of BlockN
    for (auto n = nStop; n < dN; ++n) {
        _mv_chunk<1, BlockK8>(a, &b[n * dK], dK, &out[n]);
    }
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
                float a0 = to_float(a[(2 * m + 0) * dK + k]);
                float a1 = to_float(a[(2 * m + 1) * dK + k]);
                float b0 = to_float(b[(2 * n + 0) * dK + k]);
                float b1 = to_float(b[(2 * n + 1) * dK + k]);
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
            out[m * dN + n] = to_bf16(_dot(&a[m * dK], &b[n * dK], dK));
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
                out[m * dN + n] = to_bf16(_dot(&a[m * dK], &b[n * dK], dK));
            }
        }
        // Handle remainder when dM is not a multiple of G1, `out[mStop:m1, n0:nStop]`
        // (note: excludes the bottom-right corner which is handled in the loop above)
        for (auto m = mStop; m < m1; ++m) {
            for (auto n = n0; n < nStop; ++n) {
                out[m * dN + n] = to_bf16(_dot(&a[m * dK], &b[n * dK], dK));
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
                sum += to_float(a[m * dK + k]) * to_float(b[n * dK + k]);
            }
            out[m * dN + n] = to_bf16(sum);
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
        acc = vmlaq_f32(acc, accu, vdupq_n_f32(to_float(bsi)));
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
        out[n] = to_bf16(_dot_lut8(a, &b[n * (dK / 8)], lut8, &bs[n * (dK / dG)], dK, dG));
    }
}

// LUT

template <uint64_t dG>
float _dot_lut(const bf16* __restrict__ a,
               const uint32_t* __restrict__ b,
               const uint8x16_t lut_lo,
               const uint8x16_t lut_hi,
               const bf16* __restrict__ bs,
               const uint64_t dK) {
    const uint8_t* __restrict__ bu8 = reinterpret_cast<const uint8_t*>(b);

    float32x4_t acc = vmovq_n_f32(0.0f);
    float32x4_t accu = vmovq_n_f32(0.0f);  // unscaled
    for (auto g = 0u; g < (dK / dG); ++g) {
#pragma unroll
        for (auto i = 0u; i < (dG / 16); ++i) {
            // Load 16x bf16 from a and 16x 4-bit indices from b
            auto k = g * dG + i * 16;
            auto ai0 = vld1q_bf16(a + k);
            auto ai1 = vld1q_bf16(a + k + 8);

            uint8x8_t biq = vld1_u8(&bu8[k / 2]);  // 16x 4-bit indices packed as 8 bytes
            uint8x8_t idx_lo = vand_u8(biq, vdup_n_u8(0x0F));
            uint8x8_t idx_hi = vshr_n_u8(biq, 4);
            uint8x16_t idx = vcombine_u8(idx_lo, idx_hi);
            uint8x16_t t_lo = vqtbl1q_u8(lut_lo, idx);
            uint8x16_t t_hi = vqtbl1q_u8(lut_hi, idx);
            bfloat16x8_t biu0 = vreinterpretq_bf16_u8(vzip1q_u8(t_lo, t_hi));
            bfloat16x8_t biu1 = vreinterpretq_bf16_u8(vzip2q_u8(t_lo, t_hi));

            accu = vbfdotq_f32(accu, ai0, biu0);
            accu = vbfdotq_f32(accu, ai1, biu1);
        }
        auto bsi = bs[g];
        acc = vmlaq_f32(acc, accu, vdupq_n_f32(to_float(bsi)));
        accu = vmovq_n_f32(0.0f);
    }
    return vaddvq_f32(acc);
}

template <uint64_t dG>
NOINLINE void mv_lut(const bf16* __restrict__ a,      // [dK]
                     const uint32_t* __restrict__ b,  // [dN * (dK/8)]
                     const bf16* __restrict__ lut,    // [16]
                     const bf16* __restrict__ bs,     // [dN * (dK/dG)]
                     const uint64_t dK,
                     const uint64_t dN,
                     bf16* __restrict__ out) {  // [dN]
    assert(dK % dG == 0);
    assert(dG % 8 == 0);

    // Load LUT into tables of low-bytes and high-bytes
    uint16x8_t lut0 = vreinterpretq_u16_bf16(vld1q_bf16(lut + 0));
    uint16x8_t lut1 = vreinterpretq_u16_bf16(vld1q_bf16(lut + 8));
    uint8x16_t lut_lo = vcombine_u8(vmovn_u16(lut0), vmovn_u16(lut1));
    uint8x16_t lut_hi = vcombine_u8(vshrn_n_u16(lut0, 8), vshrn_n_u16(lut1, 8));

#pragma omp parallel for
    for (auto n = 0ull; n < dN; ++n) {
        out[n] = to_bf16(_dot_lut<dG>(a, &b[n * (dK / 8)], lut_lo, lut_hi, &bs[n * (dK / dG)], dK));
    }
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

std::vector<bf16> randn(uint64_t n, std::default_random_engine& rng) {
    std::normal_distribution<float> dist(0.0, 1.0);
    std::vector<bf16> v(n);
    for (uint64_t i = 0; i < n; ++i) {
        v[i] = to_bf16(dist(rng));
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
        throw std::runtime_error(std::format("EXPECT_EQ size mismatch: {} != {} at {}:{}\n",
                                             expected.size(), actual.size(), file, line));
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(to_float(expected[i]) - to_float(actual[i])) >
            tol * std::abs(to_float(expected[i]))) {
            throw std::runtime_error(
                std::format("EXPECT_EQ value mismatch at index {}: {} != {} at {}:{}\n", i,
                            to_float(expected[i]), to_float(actual[i]), file, line));
        }
    }
}

double rmse_norm(const std::vector<bf16>& expected, const std::vector<bf16>& actual) {
    assert(expected.size() == actual.size());
    double diff_sq = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < expected.size(); i++) {
        auto ve = double(to_float(expected[i]));
        auto va = double(to_float(actual[i]));
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
    if (expected.size() != actual.size()) {
        throw std::runtime_error(std::format("EXPECT_CLOSE size mismatch: {} != {} at {}:{}\n",
                                             expected.size(), actual.size(), file, line));
    }
    auto error = rmse_norm(expected, actual);
    if (error > tol) {
        throw std::runtime_error(std::format("EXPECT_CLOSE failed: RMSE norm {} > {} at {}:{}\n",
                                             error, tol, file, line));
    }
}

void test_kernel_mm() {
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> sizes = {
        // dM, dK, dN
        {1, 120, 200},  //
        {1, 101, 203},  //
        {100, 200, 1},  //
        {32, 16, 64},   //
        {15, 133, 63},  //
        {10, 20, 30},   //
    };
    for (auto [dM, dK, dN] : sizes) {
        std::default_random_engine rng(100);
        auto a = randn(dM * dK, rng);
        auto b = randn(dN * dK, rng);
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
                float vi = std::abs(to_float(input[g * block_size + i]));
                max_abs = std::max(max_abs, vi);
                sum_abs += vi;
            }
            scale[g] = to_bf16((bits_per_element == 1) ? (sum_abs / float(block_size))
                                                       : (max_abs / float(v_max)));

            // Quantise
            for (auto i = 0ull; i < block_size; i++) {
                float v = to_float(input[g * block_size + i]) / to_float(scale[g]);
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
                lut[uint64_t(v - v_min)] = to_bf16(float(v));
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
                auto v = to_float(lut[uint64_t(q)]) * to_float(s);
                output[g * block_size + i] = to_bf16(v);
            }
        }
        return output;
    }

    // When using the LUT4 kernel, it's more efficient to process all of the low nibbles
    // together (contiguous across k), then all of the high nibbles together.
    //
    // Input:  [0,1], [2,3], [4,5],  [6,7],  [8,9],  [10,11], [12,13], [14,15]
    // Output: [0,8], [1,9], [2,10], [3,11], [4,12], [5,13],  [6,14],  [7,15]
    std::vector<uint8_t> permute_for_lut4() const {
        assert(bits_per_element == 4);
        std::vector<uint8_t> permuted(data.size());

        // Process each 16-element block (8 bytes)
        for (auto g = 0ull; g < data.size(); g += 8) {
            // Extract nibbles
            uint8_t elems[16];
            for (auto i = 0u; i < 8; ++i) {
                elems[2 * i] = data[g + i] & 0x0F;             // low nibbles
                elems[2 * i + 1] = (data[g + i] >> 4) & 0x0F;  // high nibbles
            }
            // Reorder and repack
            for (auto i = 0u; i < 8; ++i) {
                permuted[g + i] = elems[i] | uint8_t(elems[i + 8] << 4);
            }
        }
        return permuted;
    }
};

void test_block_quantise() {
    uint64_t dK = 64, dN = 8, bits_per_element = 4;
    // uint64_t dK = 32, dN = 1, bits_per_element = 4;
    constexpr uint64_t dG = 16;

    std::default_random_engine rng(200);
    auto a = randn(dK, rng);
    auto b = randn(dN * dK, rng);
    std::vector<bf16> original(dN), ref(dN), actual(dN);

    kernels::mv(a.data(), b.data(), dK, dN, original.data());

    auto bq = ScaledTensor::quantise_linear(b, dG, bits_per_element);
    auto br = bq.dequantise();
    kernels::mv(a.data(), br.data(), dK, dN, ref.data());

    // std::cerr << "bq: " << std::hex << std::vector<int>(bq.data.begin(), bq.data.end()) << "\n\n"
    //           << std::dec;

    auto lut8 = bq.lutN<2>();
    kernels::mv_lut8(a.data(), reinterpret_cast<const uint32_t*>(bq.data.data()),
                     reinterpret_cast<const uint32_t*>(lut8.data()), bq.scale.data(), dK, dN, dG,
                     actual.data());

    std::vector<bf16> actual4(dN);
    auto bq_data_perm = bq.permute_for_lut4();
    kernels::mv_lut<dG>(a.data(), reinterpret_cast<const uint32_t*>(bq_data_perm.data()),
                        bq.lut.data(), bq.scale.data(), dK, dN, actual4.data());

    EXPECT_CLOSE(ref, actual, 0.001);
    EXPECT_CLOSE(original, actual, 0.2);  // depends on bits_per_element
    EXPECT_CLOSE(ref, actual4, 0.001);
    EXPECT_CLOSE(original, actual4, 0.2);  // depends on bits_per_element
}

void test_all() {
    std::cerr << "### Running tests\n\n";
    auto success = true;
    for (auto test : {&test_kernel_mm, &test_block_quantise}) {
        try {
            test();
        } catch (const std::exception& e) {
            std::cerr << "--> Test failed: " << e.what() << "\n\n" << std::flush;
            success = false;
        }
    }
    if (success) {
        std::cerr << "---> Tests passed\n\n" << std::flush;
    }
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
    std::cerr << "### benchmark_memcpy\n";

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
    std::cerr << std::format("{:<25} {:>8.3f} ms {:>8.1f} GB/s\n\n",
                             std::to_string(bytes_to_copy / (1024 * 1024)) + " MB",
                             s.avg_time * 1e3, gbs);
}

void benchmark_reduce_sum() {
    std::cerr << "### benchmark_reduce_sum\n";

    const uint64_t n_elems = 16 * 1024 * 1024;
    const uint64_t copies = (1ull << 30) / (n_elems * sizeof(float));
    const uint64_t reps = 16;

    // Allocate
    std::vector<float> src(copies * n_elems, 0.5f);

    // Benchmark
    float result;
    auto s = measure_time(reps, [&](uint64_t i) {
        auto idx = i % copies;
        kernels::reduce_sum(&src[idx * n_elems], n_elems, &result);
    });
    double gbs = double(n_elems * sizeof(float)) / (s.avg_time * 1e9);
    std::cerr << std::format("{:<25} {:>8.3f} ms {:>8.1f} GB/s\n\n",
                             std::format("{} elements", n_elems), s.avg_time * 1e3, gbs);
}

void benchmark_mv() {
    std::cerr << "### benchmark_mv\n";

    const std::vector<std::tuple<uint64_t, uint64_t>> sizes = {
        {4096, 4096},
        {8192, 8192},
    };
    const uint64_t reps = 16;

    for (const auto& size : sizes) {
        auto dK = std::get<0>(size), dN = std::get<1>(size);
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
        std::cerr << std::format("{:<25} {:>8.3f} ms {:>8.1f} GB/s\n",
                                 std::format("{} x {}", dK, dN), s.avg_time * 1e3, gbs);
    }
    std::cerr << "\n";
}

void benchmark_mv_lut8() {
    std::cerr << "### benchmark_mv_lut8\n";

    const std::vector<std::tuple<uint64_t, uint64_t>> sizes = {
        {4096, 4096},
        {8192, 8192},
    };
    const uint64_t reps = 16;

    for (const auto& size : sizes) {
        auto dK = std::get<0>(size), dN = std::get<1>(size);

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
        std::cerr << std::format("{:<25} {:>8.3f} ms {:>8.1f} GB/s\n",
                                 std::format("{} x {}", dK, dN), s.avg_time * 1e3, gbs);
    }
    std::cerr << "\n";
}

void benchmark_mv_lut() {
    std::cerr << "### benchmark_mv_lut\n";

    const std::vector<std::tuple<uint64_t, uint64_t>> sizes = {
        {4096, 4096},
        {8192, 8192},
    };
    const uint64_t reps = 16;
    constexpr uint64_t dG = 64;

    for (const auto& size : sizes) {
        auto dK = std::get<0>(size), dN = std::get<1>(size);

        // Allocate
        auto copies = (1ull << 30) / (dK * dN * sizeof(bf16));
        std::vector<bf16> a(copies * dK, bf16(0.5f));
        std::vector<uint32_t> b(copies * dN * (dK / 8), 0xfedc0123);
        std::vector<bf16> bs(copies * dN * (dK / dG), bf16(0.25f));
        std::vector<bf16> out(copies * dN);
        std::vector<bf16> lut(16, bf16(10.0f));

        // Benchmark
        auto s = measure_time(reps, [&](uint64_t i) {
            auto idx = i % copies;
            kernels::mv_lut<dG>(&a[idx * dK], &b[idx * dN * (dK / 8)], &lut[0],
                                &bs[idx * dN * (dK / dG)], dK, dN, &out[idx * dN]);
        });
        double bytes = double(dK * sizeof(bf16) + dN * (dK / 8) * sizeof(uint32_t) +
                              dN * (dK / dG) * sizeof(bf16) + dN * sizeof(bf16));
        double gbs = bytes / (s.avg_time * 1e9);
        std::cerr << std::format("{:<25} {:>8.3f} ms {:>8.1f} GB/s\n",
                                 std::format("{} x {}", dK, dN), s.avg_time * 1e3, gbs);
    }
    std::cerr << "\n";
}

void benchmark_mm() {
    std::cerr << "### benchmark_mm\n";

    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> sizes = {
        // dM, dK, dN
        {1, 4096, 4096},
        {16, 4096, 4096},
        {256, 4096, 4096},
    };
    const uint64_t reps = 16;

    for (const auto& size : sizes) {
        auto dM = std::get<0>(size), dK = std::get<1>(size), dN = std::get<2>(size);

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
        std::cerr << std::format("{:<25} {:>8.3f} ms {:>8.1f} GB/s {:>8.1f} GFLOP/s\n",
                                 std::format("{} x {} x {}", dM, dK, dN), s.avg_time * 1e3, gbs,
                                 flops);
    }
    std::cerr << "\n";
}

}  // namespace benchmarks

// ----------------------------------------------------------------------------
// Driver program

int main() {
    auto threads = 1;
    // auto threads = 2 * std::thread::hardware_concurrency();

    omp_set_num_threads(int(threads));
    omp_set_schedule(omp_sched_static, 0);
    std::cerr << std::format("# Using {} threads\n\n", threads);

    tests::test_all();

    benchmarks::benchmark_memcpy();
    benchmarks::benchmark_reduce_sum();
    benchmarks::benchmark_mv();
    benchmarks::benchmark_mv_lut8();
    benchmarks::benchmark_mv_lut();
    benchmarks::benchmark_mm();

    return 0;
}
