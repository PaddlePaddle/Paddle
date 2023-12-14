// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include <immintrin.h>

#include "bit_cast.h"

class bfloat16_t {
public:
    bfloat16_t() = default;
    bfloat16_t(float f) { (*this) = f; }
    constexpr bfloat16_t(uint16_t r, bool) : raw_bits_(r) {}

    template <typename IntegerType,
            typename SFINAE = typename std::enable_if<std::is_integral<IntegerType>::value>::type>
    bfloat16_t(const IntegerType i)
        : raw_bits_ {convert_bits_of_normal_or_zero(bit_cast<uint32_t>(static_cast<float>(i)))} {}

    bfloat16_t &operator=(float f) {
        auto iraw = bit_cast<std::array<uint16_t, 2>>(f);
        switch (std::fpclassify(f)) {
            case FP_SUBNORMAL:
            case FP_ZERO:
                raw_bits_ = iraw[1];
                raw_bits_ &= 0x8000;
                break;
            case FP_INFINITE: raw_bits_ = iraw[1]; break;
            case FP_NAN:
                raw_bits_ = iraw[1];
                raw_bits_ |= 1 << 6;
                break;
            case FP_NORMAL:
                const uint32_t rounding_bias = 0x00007FFF + (iraw[1] & 0x1);
                const uint32_t int_raw = bit_cast<uint32_t>(f) + rounding_bias;
                iraw = bit_cast<std::array<uint16_t, 2>>(int_raw);
                raw_bits_ = iraw[1];
                break;
        }

        return *this;
    }

    template <typename IntegerType,
            typename SFINAE = typename std::enable_if<std::is_integral<IntegerType>::value>::type>
    bfloat16_t &operator=(const IntegerType i) {
        return (*this) = bfloat16_t {i};
    }

    operator float() const {
        std::array<uint16_t, 2> iraw = {{0, raw_bits_}};
        return bit_cast<float>(iraw);
    }

    bfloat16_t &operator+=(const float a) {
        (*this) = float {*this} + a;
        return *this;
    }

    // static void cvt_float_to_bfloat16(const float *src, bfloat16_t *dst, int size);
    // static void cvt_bfloat16_to_float(const bfloat16_t *src, float *dst, int size);
    // static void float_add_bfloat16(const float *src1, const bfloat16_t *src2, float *dst, int size);

private:
    constexpr uint16_t convert_bits_of_normal_or_zero(const uint32_t bits) {
        return uint32_t {bits + uint32_t {0x7FFFU + (uint32_t {bits >> 16} & 1U)}} >> 16;
    }

    uint16_t raw_bits_;
};

static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");

// inline void bfloat16_t::cvt_float_to_bfloat16(const float *src, bfloat16_t *dst, int size) {
//     constexpr int kStep = 16;

//     const __m512i nan = _mm512_set1_epi32(0xffff);
//     const __m512i ones = _mm512_set1_epi32(0x1);
//     const __m512i vec_bias = _mm512_set1_epi32(0x7fff);

// #if (__GNUC__ > 12) || ((__GNUC__ == 12) && (__GNUC_MINOR__ >= 3))
//     auto cvt_fp32_to_bf16 = [&](const __m512 input_vector) { return (__m256i)_mm512_cvtneps_pbh(input_vector); };
// #else
//     auto cvt_fp32_to_bf16 = [&](const __m512 input_vector) {
//         __m512i value = _mm512_castps_si512(input_vector);
//         auto mask = _mm512_cmp_ps_mask(input_vector, input_vector, _CMP_ORD_Q);
//         auto result = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
//         result = _mm512_add_epi32(result, vec_bias);
//         result = _mm512_add_epi32(result, value);
//         result = _mm512_srli_epi32(result, 16);
//         result = _mm512_mask_blend_epi32(mask, nan, result);
//         return _mm512_cvtusepi32_epi16(result);
//     };
// #endif

//     int blockSize = size / kStep;
//     int remainder = size % kStep;

//     for (int i = 0; i < blockSize; ++i) {
//         __m512 input_vector = _mm512_loadu_ps(src + i * kStep);
//         __m256i output_vector = cvt_fp32_to_bf16(input_vector);
//         _mm256_mask_storeu_epi16(dst + i * kStep, 0xffff, output_vector);
//     }

//     if (remainder != 0) {
//         __mmask16 mask = 0xFFFF >> (kStep - remainder);
//         __m512 input_vector = _mm512_maskz_loadu_ps(mask, src + size - remainder);
//         __m256i output_vector = cvt_fp32_to_bf16(input_vector);
//         _mm256_mask_storeu_epi16(dst + size - remainder, mask, output_vector);
//     }
// }

// inline void bfloat16_t::cvt_bfloat16_to_float(const bfloat16_t *src, float *dst, int size) {
//     constexpr int kStep = 16;

//     auto cvt_bf16_to_fp32 = [](const __m256i src) {
//         auto y = _mm512_cvtepu16_epi32(src);
//         return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
//     };

//     int blockSize = size / kStep;
//     int remainder = size % kStep;

//     for (int i = 0; i < blockSize; ++i) {
//         __m256i input_vector = _mm256_maskz_loadu_epi16(0xFFFF, src + i * kStep);
//         __m512 output_vector = cvt_bf16_to_fp32(input_vector);
//         _mm512_storeu_ps(dst + i * kStep, output_vector);
//     }

//     if (remainder != 0) {
//         __mmask16 mask = 0xFFFF >> (kStep - remainder);
//         __m256i input_vector = _mm256_maskz_loadu_epi16(mask, src + size - remainder);
//         __m512 output_vector = cvt_bf16_to_fp32(input_vector);
//         _mm512_mask_storeu_ps(dst + size - remainder, mask, output_vector);
//     }
// }

// inline void bfloat16_t::float_add_bfloat16(const float *src1, const bfloat16_t *src2, float *dst, int size) {
//     constexpr int kStep = 16;

//     auto cvt_bf16_to_fp32 = [](const __m256i src) {
//         auto y = _mm512_cvtepu16_epi32(src);
//         return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
//     };

//     int blockSize = size / kStep;
//     int remainder = size % kStep;

//     for (int i = 0; i < blockSize; ++i) {
//         __m512 vec1 = _mm512_loadu_ps(src1 + i * kStep);
//         __m256i _t = _mm256_maskz_loadu_epi16(0xFFFF, src2 + i * kStep);
//         __m512 vec2 = cvt_bf16_to_fp32(_t);
//         _mm512_storeu_ps(dst + i * kStep, vec1 + vec2);
//     }

//     if (remainder != 0) {
//         __mmask16 mask = 0xFFFF >> (kStep - remainder);
//         __m512 vec1 = _mm512_maskz_loadu_ps(mask, src1 + size - remainder);
//         __m256i _t = _mm256_maskz_loadu_epi16(mask, src2 + size - remainder);
//         __m512 vec2 = cvt_bf16_to_fp32(_t);
//         _mm512_mask_storeu_ps(dst + size - remainder, mask, vec1 + vec2);
//     }
// }
