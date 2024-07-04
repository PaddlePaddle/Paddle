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

struct PADDLE_ALIGN(1) float8_e4m3fn {
 public:
  uint8_t x;

  // Constructors
  float8_e4m3fn() = default;
  float8_e4m3fn(const float8_e4m3fn& o) = default;
  float8_e4m3fn& operator=(const float8_e4m3fn& o) = default;
  float8_e4m3fn(float8_e4m3fn&& o) = default;
  float8_e4m3fn& operator=(float8_e4m3fn&& o) = default;
  ~float8_e4m3fn() = default;

  HOSTDEVICE inline float8_e4m3fn(float val) {
#if defined(PADDLE_CUDA_FP8)
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(val);
    x = *reinterpret_cast<uint8_t*>(&tmp);
#else
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

#if defined(PADDLE_CUDA_FP8)
  HOSTDEVICE inline explicit float8_e4m3fn(const __nv_fp8_e4m3& val) {
    x = *reinterpret_cast<const uint8_t*>(&val);  // NOLINT
  }
#endif

  template <class T>
  HOSTDEVICE inline explicit float8_e4m3fn(const T& val)
      : x(float8_e4m3fn(static_cast<float>(val)).x) {}

// Assignment operators
#if defined(PADDLE_CUDA_FP8)
  HOSTDEVICE inline float8_e4m3fn& operator=(const __nv_fp8_e4m3& val) {
    x = *reinterpret_cast<const uint8_t*>(&val);  // NOLINT
    return *this;
  }
#endif

  HOSTDEVICE inline float8_e4m3fn& operator=(int8_t val) {
    x = float8_e4m3fn(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fn& operator=(uint8_t val) {
    x = float8_e4m3fn(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fn& operator=(int16_t val) {
    x = float8_e4m3fn(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fn& operator=(uint16_t val) {
    x = float8_e4m3fn(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fn& operator=(int32_t val) {
    x = float8_e4m3fn(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fn& operator=(uint32_t val) {
    x = float8_e4m3fn(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fn& operator=(int64_t val) {
    x = float8_e4m3fn(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fn& operator=(uint64_t val) {
    x = float8_e4m3fn(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fn& operator=(float val) {
    x = float8_e4m3fn(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fn& operator=(double val) {
    x = float8_e4m3fn(val).x;
    return *this;
  }

  // Conversion operators
  HOSTDEVICE inline operator float() const {
#ifdef PADDLE_CUDA_FP8
    return static_cast<float>(
        *reinterpret_cast<const __nv_fp8_e4m3*>(&x));  // NOLINT
#else
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

#ifdef PADDLE_CUDA_FP8
  HOSTDEVICE inline __nv_fp8_e4m3 to_nv_fp8_e4m3() const {
    return *reinterpret_cast<const __nv_fp8_e4m3*>(&x);
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

HOSTDEVICE inline float8_e4m3fn operator+(const float8_e4m3fn& a,
                                          const float8_e4m3fn& b) {
  return float8_e4m3fn(static_cast<float>(a) + static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3fn operator-(const float8_e4m3fn& a,
                                          const float8_e4m3fn& b) {
  return float8_e4m3fn(static_cast<float>(a) - static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3fn operator*(const float8_e4m3fn& a,
                                          const float8_e4m3fn& b) {
  return float8_e4m3fn(static_cast<float>(a) * static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3fn operator/(const float8_e4m3fn& a,
                                          const float8_e4m3fn& b) {
  return float8_e4m3fn(static_cast<float>(a) / static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3fn operator-(const float8_e4m3fn& a) {
  return float8_e4m3fn(-static_cast<float>(a));
}

HOSTDEVICE inline float8_e4m3fn& operator+=(float8_e4m3fn& a,  // NOLINT
                                            const float8_e4m3fn& b) {
  a = float8_e4m3fn(static_cast<float>(a) + static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3fn& operator-=(float8_e4m3fn& a,  // NOLINT
                                            const float8_e4m3fn& b) {
  a = float8_e4m3fn(static_cast<float>(a) - static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3fn& operator*=(float8_e4m3fn& a,  // NOLINT
                                            const float8_e4m3fn& b) {
  a = float8_e4m3fn(static_cast<float>(a) * static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3fn& operator/=(float8_e4m3fn& a,  // NOLINT
                                            const float8_e4m3fn& b) {
  a = float8_e4m3fn(static_cast<float>(a) / static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3fn raw_uint8_to_float8_e4m3fn(uint8_t a) {
  float8_e4m3fn res;
  res.x = a;
  return res;
}

// Comparison operators
HOSTDEVICE inline bool operator==(const float8_e4m3fn& a,
                                  const float8_e4m3fn& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

HOSTDEVICE inline bool operator!=(const float8_e4m3fn& a,
                                  const float8_e4m3fn& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

HOSTDEVICE inline bool operator<(const float8_e4m3fn& a,
                                 const float8_e4m3fn& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

HOSTDEVICE inline bool operator<=(const float8_e4m3fn& a,
                                  const float8_e4m3fn& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

HOSTDEVICE inline bool operator>(const float8_e4m3fn& a,
                                 const float8_e4m3fn& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

HOSTDEVICE inline bool operator>=(const float8_e4m3fn& a,
                                  const float8_e4m3fn& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

HOSTDEVICE inline bool(isnan)(const float8_e4m3fn& a) {
  return (a.x & 0b01111111) == 0b01111111;
}

// TODO(Wanglongzhi2001): According to https://arxiv.org/pdf/2209.05433.pdf,
// inf in float8_e4m3fn is undefined, to avoid compilation errors,
// we use 0b01111000 to represent inf for now!
HOSTDEVICE inline bool(isinf)(const float8_e4m3fn& a) {
  return (a.x & 0b01111111) == 0b01111000;
}

HOSTDEVICE inline bool(isfinite)(const float8_e4m3fn& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

HOSTDEVICE inline float8_e4m3fn(abs)(const float8_e4m3fn& a) {
  return float8_e4m3fn(std::abs(static_cast<float>(a)));
}

inline std::ostream& operator<<(std::ostream& os, const float8_e4m3fn& a) {
  os << static_cast<float>(a);
  return os;
}

}  // namespace dtype
}  // namespace phi

namespace std {

template <>
struct is_pod<phi::dtype::float8_e4m3fn> {
  static const bool value =
      is_trivial<phi::dtype::float8_e4m3fn>::value &&
      is_standard_layout<phi::dtype::float8_e4m3fn>::value;
};

template <>
struct is_floating_point<phi::dtype::float8_e4m3fn>
    : std::integral_constant<
          bool,
          std::is_same<phi::dtype::float8_e4m3fn,
                       typename std::remove_cv<
                           phi::dtype::float8_e4m3fn>::type>::value> {};
template <>
struct is_signed<phi::dtype::float8_e4m3fn> {
  static const bool value = true;
};

template <>
struct is_unsigned<phi::dtype::float8_e4m3fn> {
  static const bool value = false;
};

inline bool isnan(const phi::dtype::float8_e4m3fn& a) {
  return phi::dtype::isnan(a);
}

inline bool isinf(const phi::dtype::float8_e4m3fn& a) {
  return phi::dtype::isinf(a);
}

template <>
struct numeric_limits<phi::dtype::float8_e4m3fn> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 4;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 3;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -5;
  static constexpr int min_exponent10 = -1;
  static constexpr int max_exponent = 8;
  static constexpr int max_exponent10 = 2;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = false;

  HOSTDEVICE static phi::dtype::float8_e4m3fn(min)() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x08);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn lowest() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0xFE);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn(max)() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x7E);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn epsilon() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x20);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn round_error() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x30);
  }

  HOSTDEVICE static phi::dtype::float8_e4m3fn infinity() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x7F);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn quiet_NaN() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x7F);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fn denorm_min() {
    return phi::dtype::raw_uint8_to_float8_e4m3fn(0x01);
  }
};

}  // namespace std
