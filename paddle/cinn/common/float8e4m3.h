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

#if (defined(__CUDACC__) || defined(__CUDACC_RTC__)) && CUDA_VERSION >= 11800
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

  __host__ __device__ inline explicit float8e4m3(float val) {
#if defined(CINN_CUDA_FP8)
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(val);
    x = *reinterpret_cast<uint8_t*>(&tmp);
#else
    // Custom implementation for non-CUDA platforms
    // (Simplified conversion, real implementation would need proper rounding)
    float abs_val = std::abs(val);
    float scale = std::ldexp(1.0f, 7); // E4M3 max exponent
    float scaled = abs_val / scale;
    x = static_cast<uint8_t>(scaled * 127.0f);
    if (val < 0) x |= 0x80; // Set sign bit
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
    // Custom implementation for non-CUDA platforms
    bool sign = x & 0x80;
    uint8_t exp_mant = x & 0x7F;
    float val = static_cast<float>(exp_mant) / 127.0f;
    if (sign) val = -val;
    return val * std::ldexp(1.0f, 7); // Scale back
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
struct CINN_ALIGN(8) float8e4m3x4 {
  float8e4m3 x, y, z, w;
};

struct CINN_ALIGN(4) float8e4m3x2 {
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
