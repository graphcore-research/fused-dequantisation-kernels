// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

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
#include <arm_sve.h>
#include <omp.h>

#define NOINLINE __attribute__((noinline))
using bf16 = __bf16;
using i8 = int8_t;

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

// ----------------------------------------------------------------------------
// mv_naive

NOINLINE void mv_naive(const bf16* __restrict__ a,  // [dK]
                       const bf16* __restrict__ b,  // [dN * dK]
                       const uint64_t dK,
                       const uint64_t dN,
                       bf16* __restrict__ out) {  // [dN]
    for (auto n = 0u; n < dN; ++n) {
        auto sum = 0.0f;
        for (auto k = 0u; k < dK; ++k) {
            sum += to_float(a[k]) * to_float(b[n * dK + k]);
        }
        out[n] = to_bf16(sum);
    }
}

// ----------------------------------------------------------------------------
// mv

// Matrix-vector product between `a` [dK] vector and `b` [BlockN x dK] matrix
// => `out` [BlockN] vector
template <uint64_t BlockN, uint64_t BlockK>
void _mv_chunk(const bf16* __restrict__ a,
               const bf16* __restrict__ b,
               const uint64_t dK,
               bf16* __restrict__ out) {
    // Initialize accumulators
    float32x4_t accs[BlockN * BlockK];
#pragma unroll
    for (auto i = 0u; i < BlockN * BlockK; ++i) {
        accs[i] = vmovq_n_f32(0.0f);
    }
    // Main loop, process [BlockN, BlockK * 8] elements of `b` per iteration
    constexpr auto StrideK = BlockK * 8;
    const auto kStop = (dK / StrideK) * StrideK;
    for (auto k0 = 0u; k0 < kStop; k0 += StrideK) {
#pragma unroll
        for (auto iK = 0u; iK < BlockK; ++iK) {
            auto k = k0 + iK * 8;
            auto ai = vld1q_bf16(&a[k]);
#pragma unroll
            for (auto n = 0u; n < BlockN; ++n) {
                auto bi = vld1q_bf16(&b[n * dK + k]);
                auto& acc = accs[n * BlockK + iK];
                acc = vbfdotq_f32(acc, ai, bi);
            }
        }
    }
    // Accumulate partials and store results
#pragma unroll
    for (auto n = 0u; n < BlockN; ++n) {
        // Sum across BlockK accumulators
        auto& acc_n = accs[n * BlockK];
#pragma unroll
        for (auto iK = 1u; iK < BlockK; ++iK) {
            acc_n = vaddq_f32(acc_n, accs[n * BlockK + iK]);
        }
        auto sum = vaddvq_f32(acc_n);

        // Handle remainder when dK is not a multiple of StrideK
        for (auto k = kStop; k < dK; ++k) {
            sum += to_float(a[k]) * to_float(b[n * dK + k]);
        }
        out[n] = to_bf16(sum);
    }
}

template <uint64_t BlockN = 8ull, uint64_t BlockK = 2ull>
NOINLINE void mv(const bf16* __restrict__ a,  // [dK]
                 const bf16* __restrict__ b,  // [dN * dK]
                 const uint64_t dK,
                 const uint64_t dN,
                 bf16* __restrict__ out) {  // [dN]
    const auto nStop = BlockN * (dN / BlockN);
#pragma omp parallel for
    for (auto n = 0ull; n < nStop; n += BlockN) {
        _mv_chunk<BlockN, BlockK>(a, &b[n * dK], dK, &out[n]);
    }
    // Handle remainder when dN is not a multiple of BlockN
    for (auto n = nStop; n < dN; ++n) {
        _mv_chunk<1, BlockK>(a, &b[n * dK], dK, &out[n]);
    }
}

// -----------------------------------------------------------------------------
// mv_lut

// Matrix-vector product between `a` [dK] and `b` [BlockN x dK], which is 4-bit quantized,
// scaled by `bs` [BlockN x (dK/dG)] => `out` [BlockN]
template <uint64_t dG, uint64_t BlockN, uint64_t BlockK>
void _mv_lut_chunk(const bf16* __restrict__ a,
                   const uint8_t* __restrict__ b,
                   const uint8x16_t lut_lo,
                   const uint8x16_t lut_hi,
                   const bf16* __restrict__ bs,
                   const uint64_t dK,
                   bf16* __restrict__ out) {
    static_assert(dG % (BlockK * 32) == 0, "dG must be multiple of BlockK * 32");

    // Accumulators: BlockN rows x BlockK16 partial sums (unscaled within group)
    // Plus BlockN scaled accumulators for final results
    float32x4_t acc_scaled[BlockN];
#pragma unroll
    for (auto n = 0u; n < BlockN; ++n) {
        acc_scaled[n] = vmovq_n_f32(0.0f);
    }

    // Main loop over groups
    for (auto g = 0u; g < (dK / dG); ++g) {
        // Initialise unscaled accumulators
        float32x4_t accs[BlockN * BlockK];
#pragma unroll
        for (auto i = 0u; i < BlockN * BlockK; ++i) {
            accs[i] = vmovq_n_f32(0.0f);
        }
        // Accumulate over the group
#pragma unroll
        for (auto iK = 0u; iK < dG / 32; ++iK) {
            auto k = g * dG + iK * 32;
            bfloat16x8_t ai0 = vld1q_bf16(&a[k]);
            bfloat16x8_t ai1 = vld1q_bf16(&a[k + 8]);
            bfloat16x8_t ai2 = vld1q_bf16(&a[k + 16]);
            bfloat16x8_t ai3 = vld1q_bf16(&a[k + 24]);
#pragma unroll
            for (auto n = 0u; n < BlockN; ++n) {
                uint8x16_t biq = vld1q_u8(&b[n * (dK / 2) + k / 2]);
                uint8x16_t idx0 = vandq_u8(biq, vdupq_n_u8(0x0F));
                uint8x16_t idx1 = vshrq_n_u8(biq, 4);

                // Decode idx0
                uint8x16_t t0_lo = vqtbl1q_u8(lut_lo, idx0);
                uint8x16_t t0_hi = vqtbl1q_u8(lut_hi, idx0);
                bfloat16x8_t biu0 = vreinterpretq_bf16_u8(vzip1q_u8(t0_lo, t0_hi));
                bfloat16x8_t biu1 = vreinterpretq_bf16_u8(vzip2q_u8(t0_lo, t0_hi));

                // Decode idx1
                uint8x16_t t1_lo = vqtbl1q_u8(lut_lo, idx1);
                uint8x16_t t1_hi = vqtbl1q_u8(lut_hi, idx1);
                bfloat16x8_t biu2 = vreinterpretq_bf16_u8(vzip1q_u8(t1_lo, t1_hi));
                bfloat16x8_t biu3 = vreinterpretq_bf16_u8(vzip2q_u8(t1_lo, t1_hi));

                float32x4_t& acc = accs[n * BlockK + (iK % BlockK)];
                acc = vbfdotq_f32(acc, ai0, biu0);
                acc = vbfdotq_f32(acc, ai1, biu1);
                acc = vbfdotq_f32(acc, ai2, biu2);
                acc = vbfdotq_f32(acc, ai3, biu3);
            }
        }

        // Sum across BlockK accumulators and apply scale
#pragma unroll
        for (auto n = 0u; n < BlockN; ++n) {
            auto& acc_n = accs[n * BlockK];
#pragma unroll
            for (auto iK = 1u; iK < BlockK; ++iK) {
                acc_n = vaddq_f32(acc_n, accs[n * BlockK + iK]);
                accs[n * BlockK + iK] = vmovq_n_f32(0.0f);
            }
            auto scale = vcvt_f32_bf16(vld1_dup_bf16(&bs[n * (dK / dG) + g]));
            acc_scaled[n] = vmlaq_f32(acc_scaled[n], acc_n, scale);
        }
    }

    // Store results
#pragma unroll
    for (auto n = 0u; n < BlockN; ++n) {
        out[n] = to_bf16(vaddvq_f32(acc_scaled[n]));
    }
}

template <uint64_t dG, uint64_t BlockN = 8, uint64_t BlockK = 1>
NOINLINE void mv_lut(const bf16* __restrict__ a,     // [dK]
                     const uint8_t* __restrict__ b,  // [dN * (dK/2)]
                     const bf16* __restrict__ lut,   // [16]
                     const bf16* __restrict__ bs,    // [dN * (dK/dG)]
                     const uint64_t dK,
                     const uint64_t dN,
                     bf16* __restrict__ out) {  // [dN]
    assert(dK % dG == 0);
    assert(dG % 32 == 0);

    // Load LUT into tables of low-bytes and high-bytes
    uint16x8_t lut0 = vreinterpretq_u16_bf16(vld1q_bf16(&lut[0]));
    uint16x8_t lut1 = vreinterpretq_u16_bf16(vld1q_bf16(&lut[8]));
    uint8x16_t lut_lo = vcombine_u8(vmovn_u16(lut0), vmovn_u16(lut1));
    uint8x16_t lut_hi = vcombine_u8(vshrn_n_u16(lut0, 8), vshrn_n_u16(lut1, 8));

    const auto nStop = BlockN * (dN / BlockN);
#pragma omp parallel for
    for (auto n = 0ull; n < nStop; n += BlockN) {
        _mv_lut_chunk<dG, BlockN, BlockK>(a, &b[n * (dK / 2)], lut_lo, lut_hi, &bs[n * (dK / dG)],
                                          dK, &out[n]);
    }
    // Handle remainder when dN is not a multiple of BlockN
    for (auto n = nStop; n < dN; ++n) {
        _mv_lut_chunk<dG, 1, BlockK>(a, &b[n * (dK / 2)], lut_lo, lut_hi, &bs[n * (dK / dG)], dK,
                                     &out[n]);
    }
}

// ----------------------------------------------------------------------------
// mvi8_naive

NOINLINE void mvi8_naive(const i8* __restrict__ a,     // [dK]
                         const i8* __restrict__ b,     // [dN * dK]
                         const bf16* __restrict__ as,  // [1]
                         const bf16* __restrict__ bs,  // [dN]
                         const uint64_t dK,
                         const uint64_t dN,
                         bf16* __restrict__ out) {  // [dN]
    for (auto n = 0u; n < dN; ++n) {
        int32_t sum = 0;
        for (auto k = 0u; k < dK; ++k) {
            sum += int32_t(a[k]) * int32_t(b[n * dK + k]);
        }
        out[n] = to_bf16(float(sum) * to_float(as[0]) * to_float(bs[n]));
    }
}

// ----------------------------------------------------------------------------
// mvi8

template <uint64_t BlockN, uint64_t BlockK>
void _mvi8_chunk(const i8* __restrict__ a,
                 const i8* __restrict__ b,
                 const float a_scale,
                 const bf16* __restrict__ bs,
                 const uint64_t dK,
                 bf16* __restrict__ out) {
    constexpr auto StrideK = BlockK * 16;

    // Accumulators: BlockN rows x BlockK partial sums
    int32x4_t accs[BlockN * BlockK];
#pragma unroll
    for (auto i = 0u; i < BlockN * BlockK; ++i) {
        accs[i] = vdupq_n_s32(0);
    }

    // Main loop, process [BlockN, BlockK * 16] elements of `b` per iteration
    const auto kStop = (dK / StrideK) * StrideK;
    for (auto k0 = 0u; k0 < kStop; k0 += StrideK) {
#pragma unroll
        for (auto iK = 0u; iK < BlockK; ++iK) {
            const auto k = k0 + iK * 16;
            int8x16_t ai = vld1q_s8(&a[k]);
#pragma unroll
            for (auto n = 0u; n < BlockN; ++n) {
                int8x16_t bi = vld1q_s8(&b[n * dK + k]);
                int32x4_t& acc = accs[n * BlockK + iK];
                acc = vdotq_s32(acc, ai, bi);
            }
        }
    }

    // Accumulate partials and store results
#pragma unroll
    for (auto n = 0u; n < BlockN; ++n) {
        // Sum across BlockK accumulators
        int32x4_t& acc_n = accs[n * BlockK];
#pragma unroll
        for (auto iK = 1u; iK < BlockK; ++iK) {
            acc_n = vaddq_s32(acc_n, accs[n * BlockK + iK]);
        }
        int32_t sum = vaddvq_s32(acc_n);

        // Handle remainder when dK is not a multiple of StrideK
        for (auto k = kStop; k < dK; ++k) {
            sum += int32_t(a[k]) * int32_t(b[n * dK + k]);
        }

        // Store
        out[n] = to_bf16(float(sum) * a_scale * to_float(bs[n]));
    }
}

template <uint64_t BlockN = 8ull, uint64_t BlockK = 2ull>
NOINLINE void mvi8(const i8* __restrict__ a,     // [dK]
                   const i8* __restrict__ b,     // [dN * dK]
                   const bf16* __restrict__ as,  // [1]
                   const bf16* __restrict__ bs,  // [dN]
                   const uint64_t dK,
                   const uint64_t dN,
                   bf16* __restrict__ out) {  // [dN]
    const auto a_scale = to_float(as[0]);
    const auto nStop = BlockN * (dN / BlockN);
#pragma omp parallel for
    for (auto n = 0ull; n < nStop; n += BlockN) {
        _mvi8_chunk<BlockN, BlockK>(a, &b[n * dK], a_scale, &bs[n], dK, &out[n]);
    }
    for (auto n = nStop; n < dN; ++n) {
        _mvi8_chunk<1, BlockK>(a, &b[n * dK], a_scale, &bs[n], dK, &out[n]);
    }
}

// ----------------------------------------------------------------------------
// mvi8_lut3

// b is packed with 3 values per byte
// - each of the 3 values correspond to different rows (along dN)
// - each byte is a 5-bit index into lut, then a 3-bit sign vector for the 3 values
template <uint64_t BlockN, uint64_t BlockK>
void _mvi8_lut3_chunk(const i8* __restrict__ a,
                      const uint8_t* __restrict__ b,
                      const int8x16x4_t* __restrict__ lut_tables,
                      const i8* __restrict__ lut,
                      const float a_scale,
                      const bf16* __restrict__ bs,
                      const uint64_t dK,
                      bf16* __restrict__ out) {
    constexpr auto RowsPerPack = 3u;
    constexpr auto StrideK = BlockK * 16;

    int32x4_t accs[BlockN * RowsPerPack * BlockK];
#pragma unroll
    for (auto i = 0u; i < BlockN * RowsPerPack * BlockK; ++i) {
        accs[i] = vdupq_n_s32(0);
    }

    const auto k_stop = (dK / StrideK) * StrideK;
    for (auto k0 = 0u; k0 < k_stop; k0 += StrideK) {
#pragma unroll
        for (auto iK = 0u; iK < BlockK; ++iK) {
            const auto k = k0 + iK * 16;
            int8x16_t ai = vld1q_s8(&a[k]);
#pragma unroll
            for (auto n = 0u; n < BlockN; ++n) {
                uint8x16_t packed = vld1q_u8(&b[n * dK + k]);
                uint8x16_t idx0 = vandq_u8(packed, vdupq_n_u8(0x3F));
                uint8x16_t idx1 = vandq_u8(vshrq_n_u8(packed, 1), vdupq_n_u8(0x3F));
                uint8x16_t idx2 = veorq_u8(idx0, vshrq_n_u8(packed, 7));
                {  // row=0
                    int32x4_t& acc = accs[(n * RowsPerPack + 0) * BlockK + iK];
                    acc = vdotq_s32(acc, ai, vqtbl4q_s8(lut_tables[0], idx0));
                }
                {  // row=1
                    int32x4_t& acc = accs[(n * RowsPerPack + 1) * BlockK + iK];
                    acc = vdotq_s32(acc, ai, vqtbl4q_s8(lut_tables[1], idx1));
                }
                {  // row=2
                    int32x4_t& acc = accs[(n * RowsPerPack + 2) * BlockK + iK];
                    acc = vdotq_s32(acc, ai, vqtbl4q_s8(lut_tables[2], idx2));
                }
            }
        }
    }

#pragma unroll
    for (auto iN = 0u; iN < BlockN; ++iN) {
#pragma unroll
        for (auto row = 0u; row < RowsPerPack; ++row) {
            const auto n = iN * RowsPerPack + row;

            // Sum across BlockK accumulators for this row
            int32x4_t& acc_n = accs[(iN * RowsPerPack + row) * BlockK];
#pragma unroll
            for (auto iK = 1u; iK < BlockK; ++iK) {
                acc_n = vaddq_s32(acc_n, accs[(iN * RowsPerPack + row) * BlockK + iK]);
            }
            int32_t sum = vaddvq_s32(acc_n);

            // Handle remainder when dK is not a multiple of StrideK
            for (auto k = k_stop; k < dK; ++k) {
                uint8_t packed = b[iN * dK + k];
                uint8_t idx;
                if (row == 0) {
                    idx = packed & 0x3F;
                } else if (row == 1) {
                    idx = (packed >> 1) & 0x3F;
                } else {
                    idx = (packed >> 7) ^ (packed & 0x3F);
                }
                sum += int32_t(a[k]) * lut[row * 64 + idx];
            }

            // Store
            out[n] = to_bf16(float(sum) * a_scale * to_float(bs[n]));
        }
    }
}

template <uint64_t BlockN = 1ull, uint64_t BlockK = 2ull>
NOINLINE void mvi8_lut3(const i8* __restrict__ a,       // [dK]
                        const uint8_t* __restrict__ b,  // packed tensor [(dN / 3) * dK]
                        const i8* __restrict__ lut,     // [3 * 64]
                        const bf16* __restrict__ as,    // [1]
                        const bf16* __restrict__ bs,    // [dN]
                        const uint64_t dK,
                        const uint64_t dN,         // padded to multiple of 3
                        bf16* __restrict__ out) {  // [dN]
    constexpr auto RowsPerPack = 3ull;
    constexpr auto BlockRows = RowsPerPack * BlockN;
    const auto a_scale = to_float(as[0]);

    const std::array<int8x16x4_t, RowsPerPack> lut_tables = {
        int8x16x4_t{vld1q_s8(&lut[0]), vld1q_s8(&lut[16]), vld1q_s8(&lut[32]), vld1q_s8(&lut[48])},
        int8x16x4_t{vld1q_s8(&lut[64]), vld1q_s8(&lut[80]), vld1q_s8(&lut[96]),
                    vld1q_s8(&lut[112])},
        int8x16x4_t{vld1q_s8(&lut[128]), vld1q_s8(&lut[144]), vld1q_s8(&lut[160]),
                    vld1q_s8(&lut[176])},
    };
    const auto n_stop = BlockRows * (dN / BlockRows);
#pragma omp parallel for
    for (auto n = 0ull; n < n_stop; n += BlockRows) {
        _mvi8_lut3_chunk<BlockN, BlockK>(a, &b[(n / RowsPerPack) * dK], lut_tables.data(), lut,
                                         a_scale, &bs[n], dK, &out[n]);
    }
    for (auto n = n_stop; n < dN; n += RowsPerPack) {
        _mvi8_lut3_chunk<1, BlockK>(a, &b[(n / RowsPerPack) * dK], lut_tables.data(), lut, a_scale,
                                    &bs[n], dK, &out[n]);
    }
}

// ----------------------------------------------------------------------------
// mvi8_lut2

template <uint64_t BlockN, uint64_t BlockK>
void _mvi8_lut2_chunk(const i8* __restrict__ a,
                      const uint8_t* __restrict__ b,
                      const int8x16x4_t* __restrict__ lut_tables,
                      const i8* __restrict__ lut,
                      const float a_scale,
                      const bf16* __restrict__ bs,
                      const uint64_t dK,
                      bf16* __restrict__ out) {
    constexpr auto RowsPerPack = 8u;
    constexpr auto StrideK = BlockK * 16;

    int32x4_t accs[BlockN * RowsPerPack * BlockK];
#pragma unroll
    for (auto i = 0u; i < BlockN * RowsPerPack * BlockK; ++i) {
        accs[i] = vdupq_n_s32(0);
    }

    const auto k_stop = (dK / StrideK) * StrideK;
    for (auto k0 = 0u; k0 < k_stop; k0 += StrideK) {
#pragma unroll
        for (auto iK = 0u; iK < BlockK; ++iK) {
            const auto k = k0 + iK * 16;
            int8x16_t ai = vld1q_s8(&a[k]);
#pragma unroll
            for (auto iN = 0u; iN < BlockN; ++iN) {
                const auto packed = vld3q_u8(&b[iN * 3 * dK + 3 * k]);
                const auto idx0 = vandq_u8(packed.val[0], vdupq_n_u8(0x3F));
                const auto idx1 =
                    vorrq_u8(vshrq_n_u8(packed.val[0], 6),
                             vshlq_n_u8(vandq_u8(packed.val[1], vdupq_n_u8(0x0F)), 2));
                const auto idx2 =
                    vorrq_u8(vshrq_n_u8(packed.val[1], 4),
                             vshlq_n_u8(vandq_u8(packed.val[2], vdupq_n_u8(0x03)), 4));
                const auto idx3 = vshrq_n_u8(packed.val[2], 2);
                const auto idxs = std::array<uint8x16_t, 4>{idx0, idx1, idx2, idx3};

#pragma unroll
                for (auto jN = 0u; jN < RowsPerPack; ++jN) {
                    int32x4_t& acc = accs[(iN * RowsPerPack + jN) * BlockK + iK];
                    acc = vdotq_s32(acc, ai, vqtbl4q_s8(lut_tables[jN % 2], idxs[jN / 2]));
                }
            }
        }
    }

#pragma unroll
    for (auto iN = 0u; iN < BlockN; ++iN) {
#pragma unroll
        for (auto row = 0u; row < RowsPerPack; ++row) {
            const auto n = iN * RowsPerPack + row;

            // Sum across BlockK accumulators for this row
            int32x4_t& acc_n = accs[(iN * RowsPerPack + row) * BlockK];
#pragma unroll
            for (auto iK = 1u; iK < BlockK; ++iK) {
                acc_n = vaddq_s32(acc_n, accs[(iN * RowsPerPack + row) * BlockK + iK]);
            }
            int32_t sum = vaddvq_s32(acc_n);

            // Handle remainder when dK is not a multiple of StrideK
            for (auto k = k_stop; k < dK; ++k) {
                const auto b0 = b[iN * 3 * dK + 3 * k + 0];
                const auto b1 = b[iN * 3 * dK + 3 * k + 1];
                const auto b2 = b[iN * 3 * dK + 3 * k + 2];
                const auto pair = row / 2;
                uint8_t idx;
                if (pair == 0) {
                    idx = b0 & 0x3F;
                } else if (pair == 1) {
                    idx = uint8_t((b0 >> 6) | ((b1 & 0x0F) << 2));
                } else if (pair == 2) {
                    idx = uint8_t((b1 >> 4) | ((b2 & 0x03) << 4));
                } else {
                    idx = uint8_t(b2 >> 2);
                }
                sum += int32_t(a[k]) * int32_t(lut[(row % 2) * 64u + idx]);
            }

            // Store
            out[n] = to_bf16(float(sum) * a_scale * to_float(bs[n]));
        }
    }
}

template <uint64_t BlockN = 1ull, uint64_t BlockK = 1ull>
NOINLINE void mvi8_lut2(const i8* __restrict__ a,       // [dK]
                        const uint8_t* __restrict__ b,  // packed tensor [(dN / 8)*3 * dK]
                        const i8* __restrict__ lut,     // [2 * 64]
                        const bf16* __restrict__ as,    // [1]
                        const bf16* __restrict__ bs,    // [dN]
                        const uint64_t dK,
                        const uint64_t dN,         // padded to multiple of 8
                        bf16* __restrict__ out) {  // [dN]
    constexpr auto RowsPerPack = 8ull;
    constexpr auto BlockRows = RowsPerPack * BlockN;
    const auto a_scale = to_float(as[0]);

    const std::array<int8x16x4_t, 2> lut_tables = {
        int8x16x4_t{vld1q_s8(&lut[0]), vld1q_s8(&lut[16]), vld1q_s8(&lut[32]), vld1q_s8(&lut[48])},
        int8x16x4_t{vld1q_s8(&lut[64]), vld1q_s8(&lut[80]), vld1q_s8(&lut[96]),
                    vld1q_s8(&lut[112])},
    };

    const auto n_stop = BlockRows * (dN / BlockRows);
#pragma omp parallel for
    for (auto n = 0ull; n < n_stop; n += BlockRows) {
        _mvi8_lut2_chunk<BlockN, BlockK>(a, &b[(n / RowsPerPack) * 3ull * dK], lut_tables.data(),
                                         lut, a_scale, &bs[n], dK, &out[n]);
    }
    for (auto n = n_stop; n < dN; n += RowsPerPack) {
        _mvi8_lut2_chunk<1, BlockK>(a, &b[(n / RowsPerPack) * 3ull * dK], lut_tables.data(), lut,
                                    a_scale, &bs[n], dK, &out[n]);
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

void test_kernel_mv() {
    std::vector<std::tuple<uint64_t, uint64_t>> sizes = {
        // dK, dN
        {128, 64},
        {120, 200},
        {203, 101},
    };
    for (auto [dK, dN] : sizes) {
        std::default_random_engine rng(100);
        auto a = randn(dK, rng);
        auto b = randn(dN * dK, rng);
        std::vector<bf16> expected(dN);
        kernels::mv_naive(a.data(), b.data(), dK, dN, expected.data());

        std::vector<bf16> out(dN);
        kernels::mv(a.data(), b.data(), dK, dN, out.data());
        EXPECT_EQ(expected, out, 1e-3f);
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

    // It's more efficient to process all of the low nibbles together (contiguous
    // across k), then all of the high nibbles together.
    //
    // E.g. block_size=16:
    // Input:  [0,1], [2,3], [4,5],  [6,7],  [8,9],  [10,11], [12,13], [14,15]
    // Output: [0,8], [1,9], [2,10], [3,11], [4,12], [5,13],  [6,14],  [7,15]
    std::vector<uint8_t> permute_for_block_nibbles(uint64_t block_size) const {
        assert(bits_per_element == 4);
        assert(block_size % 2 == 0);
        std::vector<uint8_t> permuted(data.size());

        // Process each block
        std::vector<uint8_t> nibbles(block_size);
        for (auto g = 0ull; g < data.size(); g += (block_size / 2)) {
            // Extract nibbles
            for (auto i = 0u; i < (block_size / 2); ++i) {
                nibbles[2 * i] = data[g + i] & 0x0F;             // low nibbles
                nibbles[2 * i + 1] = (data[g + i] >> 4) & 0x0F;  // high nibbles
            }
            // Reorder and repack
            for (auto i = 0u; i < (block_size / 2); ++i) {
                permuted[g + i] = nibbles[i] | uint8_t(nibbles[i + (block_size / 2)] << 4);
            }
        }
        return permuted;
    }
};

void test_kernel_mv_lut() {
    uint64_t dK = 64, dN = 8, bits_per_element = 4;
    constexpr uint64_t dG = 32;

    std::default_random_engine rng(200);
    auto a = randn(dK, rng);
    auto b = randn(dN * dK, rng);
    std::vector<bf16> original(dN), ref(dN);

    kernels::mv(a.data(), b.data(), dK, dN, original.data());

    auto bq = ScaledTensor::quantise_linear(b, dG, bits_per_element);
    auto br = bq.dequantise();
    kernels::mv(a.data(), br.data(), dK, dN, ref.data());

    std::vector<bf16> actual(dN);
    auto bq_lut_data = bq.permute_for_block_nibbles(32);
    kernels::mv_lut<dG, 4, 1>(a.data(), bq_lut_data.data(), bq.lut.data(), bq.scale.data(), dK, dN,
                              actual.data());

    EXPECT_CLOSE(original, actual, 0.2);  // depends on bits_per_element
    EXPECT_CLOSE(ref, actual, 0.001);
}

struct ChannelI8Tensor {
    std::vector<i8> data;
    std::vector<bf16> scale;

    static ChannelI8Tensor quantise(const std::vector<bf16>& input, uint64_t dK) {
        assert(input.size() % dK == 0);
        std::vector<i8> data(input.size());
        std::vector<bf16> scale(input.size() / dK);
        for (auto n = 0ull; n < scale.size(); n++) {
            float absmax = 1e-12f;
            for (auto k = 0ull; k < dK; k++) {
                absmax = std::max(absmax, std::abs(to_float(input[n * dK + k])));
            }
            scale[n] = to_bf16(absmax / 127.0f);
            for (auto k = 0ull; k < dK; k++) {
                float v = to_float(input[n * dK + k]) / to_float(scale[n]);
                data[n * dK + k] = static_cast<i8>(std::clamp(std::round(v), -128.0f, 127.0f));
            }
        }
        return {.data = data, .scale = scale};
    }
};

void test_kernel_mvi8() {
    uint64_t dK = 128, dN = 64;
    std::default_random_engine rng(100);
    auto a = randn(dK, rng);
    auto b = randn(dN * dK, rng);
    std::vector<bf16> original(dN);
    kernels::mv_naive(a.data(), b.data(), dK, dN, original.data());

    auto ai8 = ChannelI8Tensor::quantise(a, dK);
    auto bi8 = ChannelI8Tensor::quantise(b, dK);
    std::vector<bf16> expected(dN);
    kernels::mvi8_naive(ai8.data.data(), bi8.data.data(), ai8.scale.data(), bi8.scale.data(), dK,
                        dN, expected.data());

    std::vector<bf16> actual(dN);
    kernels::mvi8(ai8.data.data(), bi8.data.data(), ai8.scale.data(), bi8.scale.data(), dK, dN,
                  actual.data());

    EXPECT_CLOSE(original, actual, 0.05);
    EXPECT_EQ(expected, actual, 1e-3f);
}

bool all_tests_passed = true;
#define RUN_TEST(test) tests::run_test(test, #test);
void run_test(const std::function<void()>& test, const std::string& test_name) {
    try {
        test();
    } catch (const std::exception& e) {
        std::cerr << "--> " << test_name << " failed: " << e.what() << "\n\n" << std::flush;
        all_tests_passed = false;
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
        std::vector<uint32_t> b32(copies * dN * (dK / 8), 0xfedc0123);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(b32.data());
        std::vector<bf16> bs(copies * dN * (dK / dG), bf16(0.25f));
        std::vector<bf16> out(copies * dN);
        std::vector<bf16> lut(16, bf16(10.0f));

        // Benchmark
        auto s = measure_time(reps, [&](uint64_t i) {
            auto idx = i % copies;
            kernels::mv_lut<dG>(&a[idx * dK], &b[idx * dN * (dK / 2)], &lut[0],
                                &bs[idx * dN * (dK / dG)], dK, dN, &out[idx * dN]);
        });
        double bytes = double(dK * sizeof(bf16) + dN * (dK / 2) * sizeof(uint8_t) +
                              dN * (dK / dG) * sizeof(bf16) + dN * sizeof(bf16));
        double gbs = bytes / (s.avg_time * 1e9);
        std::cerr << std::format("{:<25} {:>8.3f} ms {:>8.1f} GB/s\n",
                                 std::format("{} x {}", dK, dN), s.avg_time * 1e3, gbs);
    }
    std::cerr << "\n";
}

void benchmark_mvi8() {
    std::cerr << "### benchmark_mvi8\n";

    const std::vector<std::tuple<uint64_t, uint64_t>> sizes = {
        {4096, 4096},
        {8192, 8192},
    };
    const uint64_t reps = 16;

    for (const auto& size : sizes) {
        auto dK = std::get<0>(size), dN = std::get<1>(size);

        // Allocate
        auto copies = std::max<uint64_t>(1, (1ull << 30) / (dN * dK * sizeof(i8)));
        std::vector<i8> a(copies * dK, i8(64));
        std::vector<i8> b(copies * dN * dK, i8(64));
        std::vector<bf16> as(copies, bf16(0.5f / 64.0f));
        std::vector<bf16> bs(copies * dN, bf16(0.5f / 64.0f));
        std::vector<bf16> out(copies * dN);

        // Benchmark
        auto s = measure_time(reps, [&](uint64_t i) {
            auto idx = i % copies;
            kernels::mvi8(&a[idx * dK], &b[idx * dN * dK], &as[idx], &bs[idx * dN], dK, dN,
                          &out[idx * dN]);
        });
        double bytes =
            double(dK * sizeof(i8) + dN * dK * sizeof(i8) + dN * sizeof(bf16) + dN * sizeof(bf16));
        double gbs = bytes / (s.avg_time * 1e9);
        std::cerr << std::format("{:<25} {:>8.3f} ms {:>8.1f} GB/s\n",
                                 std::format("{} x {}", dK, dN), s.avg_time * 1e3, gbs);
    }
    std::cerr << "\n";
}

void benchmark_mvi8_lut3() {
    std::cerr << "### benchmark_mvi8_lut3\n";

    const std::vector<std::tuple<uint64_t, uint64_t>> sizes = {
        {4096, 4096},
        {8192, 8192},
    };
    const uint64_t reps = 16;

    for (const auto& size : sizes) {
        auto dK = std::get<0>(size), dN = std::get<1>(size);
        const auto dN_pad = ((dN + 2) / 3) * 3;

        // Allocate
        auto copies = std::max<uint64_t>(1, (1ull << 30) / ((dN_pad / 3) * dK * sizeof(uint8_t)));
        std::vector<i8> a(copies * dK, i8(64));
        std::vector<uint8_t> b(copies * dN_pad * dK, uint8_t(0));
        std::vector<i8> lut(3 * 64, i8(0));
        std::vector<bf16> as(copies, bf16(0.5f / 64.0f));
        std::vector<bf16> bs(copies * dN_pad, bf16(0.5f / 64.0f));
        std::vector<bf16> out(copies * dN_pad);
        for (auto i = 0u; i < 64u; ++i) {
            lut[0 * 64 + i] = static_cast<i8>((int(i) % 15) - 7);
            lut[1 * 64 + i] = static_cast<i8>(((int(i) + 5) % 15) - 7);
            lut[2 * 64 + i] = static_cast<i8>(((int(i) + 10) % 15) - 7);
        }
        for (auto idx = 0ull; idx < b.size(); ++idx) {
            const auto lut_index = uint8_t(idx % 32ull);
            const auto sign_bits = uint8_t(((idx / 32ull) % 8ull) << 5);
            b[idx] = uint8_t((sign_bits & 0x01) | lut_index | ((sign_bits & 0x02) << 5) |
                             ((sign_bits & 0x04) << 5));
        }

        // Benchmark
        auto s = measure_time(reps, [&](uint64_t i) {
            auto idx = i % copies;
            kernels::mvi8_lut3(&a[idx * dK], &b[idx * dN_pad * dK], lut.data(), &as[idx],
                               &bs[idx * dN_pad], dK, dN, &out[idx * dN_pad]);
        });
        double bytes = double(dK * sizeof(i8) + (dN_pad / 3) * dK * sizeof(uint8_t) +
                              dN * sizeof(bf16) + dN * sizeof(bf16));
        double gbs = bytes / (s.avg_time * 1e9);
        std::cerr << std::format("{:<25} {:>8.3f} ms {:>8.1f} GB/s\n",
                                 std::format("{} x {}", dK, dN), s.avg_time * 1e3, gbs);
    }
    std::cerr << "\n";
}

void benchmark_mvi8_lut2() {
    std::cerr << "### benchmark_mvi8_lut2\n";

    const std::vector<std::tuple<uint64_t, uint64_t>> sizes = {
        {4096, 4096},
        {8192, 8192},
    };
    const uint64_t reps = 16;

    for (const auto& size : sizes) {
        auto dK = std::get<0>(size), dN = std::get<1>(size);

        auto copies =
            std::max<uint64_t>(1, (1ull << 30) / ((dN / 8) * 3ull * dK * sizeof(uint8_t)));
        std::vector<i8> a(copies * dK, i8(64));
        std::vector<uint8_t> b(copies * (dN / 8) * 3ull * dK, uint8_t(0));
        std::vector<i8> lut(2 * 64, i8(0));
        std::vector<bf16> as(copies, bf16(0.5f / 64.0f));
        std::vector<bf16> bs(copies * dN, bf16(0.5f / 64.0f));
        std::vector<bf16> out(copies * dN);

        for (auto i = 0u; i < 64u; ++i) {
            lut[0 * 64 + i] = static_cast<i8>((int(i) % 15) - 7);
            lut[1 * 64 + i] = static_cast<i8>(((int(i) + 5) % 15) - 7);
        }
        for (auto idx = 0ull; idx < (dN / 8) * dK * copies; ++idx) {
            const auto idx0 = uint8_t((idx + 0) % 64ull);
            const auto idx1 = uint8_t((idx + 1) % 64ull);
            const auto idx2 = uint8_t((idx + 2) % 64ull);
            const auto idx3 = uint8_t((idx + 3) % 64ull);
            b[3 * idx + 0] = uint8_t((idx0 & 0x3Fu) | uint8_t((idx1 & 0x03u) << 6));
            b[3 * idx + 1] = uint8_t(((idx1 >> 2) & 0x0Fu) | uint8_t((idx2 & 0x0Fu) << 4));
            b[3 * idx + 2] = uint8_t(((idx2 >> 4) & 0x03u) | uint8_t((idx3 & 0x3Fu) << 2));
        }

        auto s = measure_time(reps, [&](uint64_t i) {
            auto idx = i % copies;
            kernels::mvi8_lut2(&a[idx * dK], &b[idx * ((dN / 8) * 3ull * dK)], lut.data(), &as[idx],
                               &bs[idx * dN], dK, dN, &out[idx * dN]);
        });
        double bytes = double(dK * sizeof(i8) + (dN / 8) * 3ull * dK * sizeof(uint8_t) +
                              dN * sizeof(bf16) + dN * sizeof(bf16));
        double gbs = bytes / (s.avg_time * 1e9);
        std::cerr << std::format("{:<25} {:>8.3f} ms {:>8.1f} GB/s\n",
                                 std::format("{} x {}", dK, dN), s.avg_time * 1e3, gbs);
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
    std::cerr << std::format("# Using {} threads\n", threads);
    std::cerr << std::format("# SVE vector length: {} bits\n", 8 * svcntb());
    std::cerr << "\n";

    RUN_TEST(tests::test_kernel_mv);
    RUN_TEST(tests::test_kernel_mv_lut);
    RUN_TEST(tests::test_kernel_mvi8);
    std::cerr << "### Tests " << (tests::all_tests_passed ? "PASSED" : "FAILED") << "\n\n";

    benchmarks::benchmark_memcpy();
    benchmarks::benchmark_mv();
    benchmarks::benchmark_mv_lut();
    benchmarks::benchmark_mvi8();
    benchmarks::benchmark_mvi8_lut3();
    benchmarks::benchmark_mvi8_lut2();

    return 0;
}
