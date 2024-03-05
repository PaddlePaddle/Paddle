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

class float16_t {
public:
    float16_t() = default;
    float16_t(float val) { (*this) = val; }
    constexpr float16_t(uint16_t bits, bool) : raw_(bits) {}

    float16_t &operator=(float val);

    float16_t &operator+=(float16_t a) {
        (*this) = float(f() + a.f());
        return *this;
    }

    operator float() const;

    static void cvt_float_to_float16(const float *src, float16_t *dst, int size);
    static void cvt_float16_to_float(const float16_t *src, float *dst, int size);
    static void float_add_float16(const float *src1, const float16_t *src2, float *dst, int size);

private:
    float f() { return (float)(*this); }

    uint16_t raw_;
};

static_assert(sizeof(float16_t) == 2, "float16_t must be 2 bytes");

inline float16_t &float16_t::operator=(float f) {
    uint32_t i = bit_cast<uint32_t>(f);
    uint32_t s = i >> 31;
    uint32_t e = (i >> 23) & 0xFF;
    uint32_t m = i & 0x7FFFFF;

    uint32_t ss = s;
    uint32_t mm = m >> 13;
    uint32_t r = m & 0x1FFF;
    uint32_t ee = 0;
    int32_t eee = (e - 127) + 15;

    if (0 == e) {
        ee = 0;
        mm = 0;
    } else if (0xFF == e) {
        ee = 0x1F;
        if (0 != m && 0 == mm) mm = 1;
    } else if (0 < eee && eee < 0x1F) {
        ee = eee;
        if (r > (0x1000 - (mm & 1))) {
            mm++;
            if (mm == 0x400) {
                mm = 0;
                ee++;
            }
        }
    } else if (0x1F <= eee) {
        ee = 0x1F;
        mm = 0;
    } else {
        float ff = fabsf(f) + 0.5;
        uint32_t ii = bit_cast<uint32_t>(ff);
        ee = 0;
        mm = ii & 0x7FF;
    }

    this->raw_ = (ss << 15) | (ee << 10) | mm;
    return *this;
}

inline float16_t::operator float() const {
    uint32_t ss = raw_ >> 15;
    uint32_t ee = (raw_ >> 10) & 0x1F;
    uint32_t mm = raw_ & 0x3FF;

    uint32_t s = ss;
    uint32_t eee = ee - 15 + 127;
    uint32_t m = mm << 13;
    uint32_t e;

    if (0 == ee) {
        if (0 == mm) {
            e = 0;
        } else {
            return (ss ? -1 : 1) * std::scalbn((float)mm, -24);
        }
    } else if (0x1F == ee) {
        e = 0xFF;
    } else {
        e = eee;
    }

    uint32_t f = (s << 31) | (e << 23) | m;

    return bit_cast<float>(f);
}

inline void float16_t::cvt_float_to_float16(const float *src, float16_t *dst, int size) {
    // Round to nearest even mode
    constexpr int rounding_mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    // Process 16 floats (AVX512 is a 512-bit SIMD register)
    constexpr int kStep = 16;
    int blockSize = size / kStep;
    int remainder = size % kStep;

    // Process blocks of 16 floats at a time
    for (int i = 0; i < blockSize; ++i) {
        // Load the input floats into a AVX512 register
        __m512 input_vector = _mm512_loadu_ps(src + i * kStep);

        // Convert the floats to float16_t using AVX512 intrinsics
        __m256i output_vector = _mm512_cvtps_ph(input_vector, rounding_mode);

        // Store the converted values in the output array
        _mm256_mask_storeu_epi16(dst + i * kStep, 0xffff, output_vector);
    }

    if (remainder != 0) {
        __mmask16 mask = 0xFFFF >> (kStep - remainder);
        __m512 input_vector = _mm512_maskz_loadu_ps(mask, src + size - remainder);
        __m256i output_vector = _mm512_cvtps_ph(input_vector, rounding_mode);
        _mm256_mask_storeu_epi16(dst + size - remainder, mask, output_vector);
    }
}

inline void float16_t::cvt_float16_to_float(const float16_t *src, float *dst, int size) {
    // Process 16 floats (AVX512 is a 512-bit SIMD register)
    constexpr int kStep = 16;
    int blockSize = size / kStep;
    int remainder = size % kStep;

    for (int i = 0; i < blockSize; ++i) {
        __m256i input_vector = _mm256_maskz_loadu_epi16(0xffff, src + i * kStep);
        __m512 output_vector = _mm512_cvtph_ps(input_vector);
        _mm512_storeu_ps(dst + i * kStep, output_vector);
    }

    if (remainder != 0) {
        __mmask16 mask = 0xFFFF >> (kStep - remainder);
        __m256i input_vector = _mm256_maskz_loadu_epi16(mask, src + size - remainder);
        __m512 output_vector = _mm512_cvtph_ps(input_vector);
        _mm512_mask_storeu_ps(dst + size - remainder, mask, output_vector);
    }
}

inline void float16_t::float_add_float16(const float *src1, const float16_t *src2, float *dst, int size) {
    constexpr int kStep = 16;
    int blockSize = size / kStep;
    int remainder = size % kStep;

    for (int i = 0; i < blockSize; ++i) {
        __m512 vec1 = _mm512_loadu_ps(src1 + i * kStep);
        __m256i _t = _mm256_maskz_loadu_epi16(0xffff, src2 + i * kStep);
        __m512 vec2 = _mm512_cvtph_ps(_t);
        _mm512_storeu_ps(dst + i * kStep, vec1 + vec2);
    }

    if (remainder != 0) {
        __mmask16 mask = 0xFFFF >> (kStep - remainder);
        __m512 vec1 = _mm512_maskz_loadu_ps(mask, src1 + size - remainder);
        __m256i _t = _mm256_maskz_loadu_epi16(mask, src2 + size - remainder);
        __m512 vec2 = _mm512_cvtph_ps(_t);
        _mm512_mask_storeu_ps(dst + size - remainder, mask, vec1 + vec2);
    }
}
