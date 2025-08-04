// Copyright (c) 2025 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CINN_COMMON_FLOAT8E4M3_H
#define CINN_COMMON_FLOAT8E4M3_H

#ifdef __cplusplus
#pragma once
#endif  // __cplusplus

#include <stdint.h>

#include <cmath>
#include <cstring>

#ifdef CINN_WITH_CUDA
#include <cuda.h>

#if (defined(__CUDACC__) || defined(__CUDACC_RTC__)) && CUDA_VERSION >= 11080
#define CINN_CUDA_FP8
#include <cuda_fp8.h>
#endif  // __CUDACC__
#endif  // CINN_WITH_CUDA

#ifdef __cplusplus

#ifndef _WIN32
#define CINN_ALIGN(x) __attribute__((aligned(x)))
#else  // _WIN32
#define CINN_ALIGN(x) __declspec(align(x))
#endif  // _WIN32

#else  // __cplusplus
#define CINN_ALIGN(x)
#endif  // __cplusplus

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#ifdef __cplusplus
namespace cinn {
namespace common {
#endif  // __cplusplus

// E4M3 format (4 exponent bits, 3 mantissa bits)
struct CINN_ALIGN(1) float8e4m3 {
  uint8_t x;

#ifdef __cplusplus
  // Constructors
  float8e4m3() = default;
  float8e4m3(const float8e4m3& o) = default;
  float8e4m3& operator=(const float8e4m3& o) = default;
  float8e4m3(float8e4m3&& o) = default;
  float8e4m3& operator=(float8e4m3&& o) = default;
  ~float8e4m3() = default;

  union Bits {
    float f;
    uint32_t ui;
  };
  __host__ __device__ inline explicit float8e4m3(float val) {
#if defined(CINN_CUDA_FP8)
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(val);
    x = *reinterpret_cast<uint8_t*>(&tmp);
#else
    // NOTE(YuhanXu): this code is mainly from
    // https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/float8_e4m3fn.h
    // with minor changes.
    // CPU implementation.
    Bits fb, denorm_mask;
    fb.f = val;
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
    denorm_mask.ui = UINT32_C(141) << 23;
    uint8_t result = 0u;
    const uint32_t sign = fb.ui & UINT32_C(0x80000000);
    fb.ui ^= sign;
    if (fb.ui >= fp8_max) {
      result = 0x7e;
    } else {
      if (fb.ui < (UINT32_C(121) << 23)) {
        fb.f = fb.f + denorm_mask.f;
        fb.ui = fb.ui - denorm_mask.ui;
        result = static_cast<uint8_t>(fb.ui);
      } else {
        uint8_t mant_odd = (fb.ui >> 20) & 1;
        fb.ui += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        fb.ui += mant_odd;
        result = static_cast<uint8_t>(fb.ui >> 20);
      }
    }

    result |= static_cast<uint8_t>(sign >> 24);
    x = result;
#endif
  }

#if defined(CINN_CUDA_FP8)
  __host__ __device__ inline explicit float8e4m3(const __nv_fp8_e4m3& val) {
    x = *reinterpret_cast<const uint8_t*>(&val);
  }
  __host__ __device__ inline explicit float8e4m3(const __nv_bfloat16& val) {
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(val);
    x = *reinterpret_cast<uint8_t*>(&tmp);
  }
#endif

  template <class T>
  __host__ __device__ inline explicit float8e4m3(const T& val)
      : x(float8e4m3(static_cast<float>(val)).x) {}

// Assignment operators
#if defined(CINN_CUDA_FP8)
  __host__ __device__ inline float8e4m3& operator=(const __nv_fp8_e4m3& val) {
    x = *reinterpret_cast<const uint8_t*>(&val);  // NOLINT
    return *this;
  }
#endif

  // Conversion operators
  __host__ __device__ inline operator float() const {
#ifdef CINN_CUDA_FP8
    return static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(&x));
#else
    // NOTE(YuhanXu): this code is mainly from
    // https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/float8_e4m3fn.h
    // with minor changes.
    // CPU implementation.
    const uint32_t w = (uint32_t)x << 24;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);

    // get the leading 0-bits in nonsin.
    uint32_t nonsign_tmp = nonsign;
    uint32_t renorm_shift = 0;
    if (nonsign_tmp == 0) {
      renorm_shift = sizeof(uint32_t) * 8;
    } else {
      if ((nonsign_tmp & 0xFFFF0000) == 0) {
        renorm_shift += 16;
        nonsign_tmp <<= 16;
      }
      if ((nonsign_tmp & 0xFF000000) == 0) {
        renorm_shift += 8;
        nonsign_tmp <<= 8;
      }
      if ((nonsign_tmp & 0xF0000000) == 0) {
        renorm_shift += 4;
        nonsign_tmp <<= 4;
      }
      if ((nonsign_tmp & 0xC0000000) == 0) {
        renorm_shift += 2;
        nonsign_tmp <<= 2;
      }
      if ((nonsign_tmp & 0x80000000) == 0) {
        renorm_shift += 1;
      }
    }

    renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
    const int32_t inf_nan_mask =
        ((int32_t)(nonsign + 0x01000000) >> 8) & INT32_C(0x7F800000);
    const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
    Bits result;
    result.ui =
        sign |
        ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23)) |
          inf_nan_mask) &
         ~zero_mask);
    return result.f;
#endif
  }

#ifdef CINN_CUDA_FP8
  __host__ __device__ inline __nv_fp8_e4m3 to_nv_fp8_e4m3() const {
    return *reinterpret_cast<const __nv_fp8_e4m3*>(&x);
  }
#endif

  __host__ __device__ inline explicit operator bool() const {
    return (x & 0x7fff) != 0;
  }

  __host__ __device__ inline explicit operator int8_t() const {
    return static_cast<int8_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

  __host__ __device__ inline operator double() const {
    return static_cast<double>(static_cast<float>(*this));
  }
#endif  // __cplusplus
};

// Vector types
struct CINN_ALIGN(4) float8e4m34 {
  float8e4m3 x, y, z, w;
};

struct CINN_ALIGN(2) float8e4m32 {
  float8e4m3 x, y;
};

#ifdef __cplusplus

/// TODO(Yuhan): Arithmetic operator+ - * / etc.

__host__ __device__ inline float8e4m3 raw_uint8_to_float8e4m3(uint8_t a) {
  float8e4m3 res;
  res.x = a;
  return res;
}

/// TODO(Yuhan): Comparison operators operator== != > < <= >= / etc.

}  // namespace common
}  // namespace cinn
#endif  // __cplusplus

#endif  // CINN_COMMON_FLOAT8E4M3_H
