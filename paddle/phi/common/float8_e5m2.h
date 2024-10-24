// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <stdint.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include "paddle/common/hostdevice.h"
#include "paddle/phi/common/float16.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif

#if defined(__CUDACC__) && CUDA_VERSION >= 11800
#define PADDLE_CUDA_FP8
#include <cuda_fp8.h>
#endif

#ifndef PADDLE_WITH_HIP
#if !defined(_WIN32)
#define PADDLE_ALIGN(x) __attribute__((aligned(x)))
#else
#define PADDLE_ALIGN(x) __declspec(align(x))
#endif
#else
#define PADDLE_ALIGN(x)
#endif

namespace phi {
namespace dtype {

struct PADDLE_ALIGN(1) float8_e5m2 {
 public:
  uint8_t x;

  // Constructors
  float8_e5m2() = default;
  float8_e5m2(const float8_e5m2& o) = default;
  float8_e5m2& operator=(const float8_e5m2& o) = default;
  float8_e5m2(float8_e5m2&& o) = default;
  float8_e5m2& operator=(float8_e5m2&& o) = default;
  ~float8_e5m2() = default;

  HOSTDEVICE inline float8_e5m2(float val) {
#if defined(PADDLE_CUDA_FP8)
    __nv_fp8_e5m2 tmp = __nv_fp8_e5m2(val);
    x = *reinterpret_cast<uint8_t*>(&tmp);
#else
    // CPU implementation.
    Bits fb, denorm_mask;
    fb.f = val;

    constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
    constexpr uint32_t fp8_max = UINT32_C(143) << 23;
    denorm_mask.ui = UINT32_C(134) << 23;

    uint8_t result = 0u;

    const uint32_t sign = fb.ui & UINT32_C(0x80000000);

    fb.ui ^= sign;

    if (fb.ui >= fp8_max) {
      result = fb.ui > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7B);
    } else {
      if (fb.ui < (UINT32_C(113) << 23)) {
        fb.f = fb.f + denorm_mask.f;
        result = static_cast<uint8_t>(fb.ui - denorm_mask.ui);
      } else {
        uint32_t mant_odd = (fb.ui >> 21) & 1;
        fb.ui += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
        fb.ui += mant_odd;
        result = static_cast<uint8_t>(fb.ui >> 21);
      }
    }

    result |= static_cast<uint8_t>(sign >> 24);
    x = result;
#endif
  }

#if defined(PADDLE_CUDA_FP8)
  HOSTDEVICE inline explicit float8_e5m2(const __nv_fp8_e5m2& val) {
    x = *reinterpret_cast<const uint8_t*>(&val);  // NOLINT
  }
#endif

  template <class T>
  HOSTDEVICE inline explicit float8_e5m2(const T& val)
      : x(float8_e5m2(static_cast<float>(val)).x) {}

// Assignment operators
#if defined(PADDLE_CUDA_FP8)
  HOSTDEVICE inline float8_e5m2& operator=(const __nv_fp8_e5m2& val) {
    x = *reinterpret_cast<const uint8_t*>(&val);  // NOLINT
    return *this;
  }
#endif

  HOSTDEVICE inline float8_e5m2& operator=(int8_t val) {
    x = float8_e5m2(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e5m2& operator=(uint8_t val) {
    x = float8_e5m2(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e5m2& operator=(int16_t val) {
    x = float8_e5m2(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e5m2& operator=(uint16_t val) {
    x = float8_e5m2(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e5m2& operator=(int32_t val) {
    x = float8_e5m2(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e5m2& operator=(uint32_t val) {
    x = float8_e5m2(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e5m2& operator=(int64_t val) {
    x = float8_e5m2(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e5m2& operator=(uint64_t val) {
    x = float8_e5m2(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e5m2& operator=(float val) {
    x = float8_e5m2(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e5m2& operator=(double val) {
    x = float8_e5m2(val).x;
    return *this;
  }

  // Conversion operators
  HOSTDEVICE inline operator float() const {
#ifdef PADDLE_CUDA_FP8
    return float(*reinterpret_cast<const __nv_fp8_e5m2*>(&x));  // NOLINT
#else
    // refer to
    // https://github.com/pytorch/pytorch/blob/main/c10/util/Float8_e5m2.h
    uint16_t half_representation = x;
    half_representation <<= 8;
    return static_cast<float>(
        phi::dtype::raw_uint16_to_float16(half_representation));
#endif
  }

#ifdef PADDLE_CUDA_FP8
  HOSTDEVICE inline __nv_fp8_e5m2 to_nv_fp8_e5m2() const {
    return *reinterpret_cast<const __nv_fp8_e5m2*>(&x);
  }
#endif

  HOSTDEVICE inline explicit operator int8_t() const {
    return static_cast<int8_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

 private:
  union Bits {
    float f;
    uint32_t ui;
  };
};

HOSTDEVICE inline float8_e5m2 operator+(const float8_e5m2& a,
                                        const float8_e5m2& b) {
  return float8_e5m2(static_cast<float>(a) + static_cast<float>(b));
}

HOSTDEVICE inline float8_e5m2 operator-(const float8_e5m2& a,
                                        const float8_e5m2& b) {
  return float8_e5m2(static_cast<float>(a) - static_cast<float>(b));
}

HOSTDEVICE inline float8_e5m2 operator*(const float8_e5m2& a,
                                        const float8_e5m2& b) {
  return float8_e5m2(static_cast<float>(a) * static_cast<float>(b));
}

HOSTDEVICE inline float8_e5m2 operator/(const float8_e5m2& a,
                                        const float8_e5m2& b) {
  return float8_e5m2(static_cast<float>(a) / static_cast<float>(b));
}

HOSTDEVICE inline float8_e5m2 operator-(const float8_e5m2& a) {
  return float8_e5m2(-static_cast<float>(a));
}

HOSTDEVICE inline float8_e5m2& operator+=(float8_e5m2& a,  // NOLINT
                                          const float8_e5m2& b) {
  a = float8_e5m2(static_cast<float>(a) + static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e5m2& operator-=(float8_e5m2& a,  // NOLINT
                                          const float8_e5m2& b) {
  a = float8_e5m2(static_cast<float>(a) - static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e5m2& operator*=(float8_e5m2& a,  // NOLINT
                                          const float8_e5m2& b) {
  a = float8_e5m2(static_cast<float>(a) * static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e5m2& operator/=(float8_e5m2& a,  // NOLINT
                                          const float8_e5m2& b) {
  a = float8_e5m2(static_cast<float>(a) / static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e5m2 raw_uint8_to_float8_e5m2(uint8_t a) {
  float8_e5m2 res;
  res.x = a;
  return res;
}

// Comparison operators
HOSTDEVICE inline bool operator==(const float8_e5m2& a, const float8_e5m2& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

HOSTDEVICE inline bool operator!=(const float8_e5m2& a, const float8_e5m2& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

HOSTDEVICE inline bool operator<(const float8_e5m2& a, const float8_e5m2& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

HOSTDEVICE inline bool operator<=(const float8_e5m2& a, const float8_e5m2& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

HOSTDEVICE inline bool operator>(const float8_e5m2& a, const float8_e5m2& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

HOSTDEVICE inline bool operator>=(const float8_e5m2& a, const float8_e5m2& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

HOSTDEVICE inline bool(isnan)(const float8_e5m2& a) {
  return (a.x & 0b01111111) > 0b01111100;
}

HOSTDEVICE inline bool(isinf)(const float8_e5m2& a) {
  return (a.x & 0b01111111) == 0b01111100;
}

HOSTDEVICE inline bool(isfinite)(const float8_e5m2& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

HOSTDEVICE inline float8_e5m2(abs)(const float8_e5m2& a) {
  return float8_e5m2(std::abs(static_cast<float>(a)));
}

inline std::ostream& operator<<(std::ostream& os, const float8_e5m2& a) {
  os << static_cast<float>(a);
  return os;
}

}  // namespace dtype
}  // namespace phi

namespace std {

template <>
struct is_pod<phi::dtype::float8_e5m2> {
  static const bool value = is_trivial<phi::dtype::float8_e5m2>::value &&
                            is_standard_layout<phi::dtype::float8_e5m2>::value;
};

template <>
struct is_floating_point<phi::dtype::float8_e5m2>
    : std::integral_constant<
          bool,
          std::is_same<
              phi::dtype::float8_e5m2,
              typename std::remove_cv<phi::dtype::float8_e5m2>::type>::value> {
};
template <>
struct is_signed<phi::dtype::float8_e5m2> {
  static const bool value = true;
};

template <>
struct is_unsigned<phi::dtype::float8_e5m2> {
  static const bool value = false;
};

inline bool isnan(const phi::dtype::float8_e5m2& a) {
  return phi::dtype::isnan(a);
}

inline bool isinf(const phi::dtype::float8_e5m2& a) {
  return phi::dtype::isinf(a);
}

template <>
struct numeric_limits<phi::dtype::float8_e5m2> {
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_specialized = true;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = false;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 3;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 2;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  HOSTDEVICE static phi::dtype::float8_e5m2(min)() {
    return phi::dtype::raw_uint8_to_float8_e5m2(0x4);
  }
  HOSTDEVICE static phi::dtype::float8_e5m2 lowest() {
    return phi::dtype::raw_uint8_to_float8_e5m2(0xFB);
  }
  HOSTDEVICE static phi::dtype::float8_e5m2(max)() {
    return phi::dtype::raw_uint8_to_float8_e5m2(0x7B);
  }
  HOSTDEVICE static phi::dtype::float8_e5m2 epsilon() {
    return phi::dtype::raw_uint8_to_float8_e5m2(0x34);
  }
  HOSTDEVICE static phi::dtype::float8_e5m2 round_error() {
    return phi::dtype::raw_uint8_to_float8_e5m2(0x38);
  }
  HOSTDEVICE static phi::dtype::float8_e5m2 infinity() {
    return phi::dtype::raw_uint8_to_float8_e5m2(0x7C);
  }
  HOSTDEVICE static phi::dtype::float8_e5m2 denorm_min() {
    return phi::dtype::raw_uint8_to_float8_e5m2(0x01);
  }
};

}  // namespace std
