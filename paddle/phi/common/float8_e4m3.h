// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/hostdevice.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif

#if defined(__CUDACC__) && CUDA_VERSION >= 12000
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

struct float8_e4m3 {
 public:
  uint8_t x;

  // Constructors
  float8_e4m3() = default;
  float8_e4m3(const float8_e4m3& o) = default;
  float8_e4m3& operator=(const float8_e4m3& o) = default;
  float8_e4m3(float8_e4m3&& o) = default;
  float8_e4m3& operator=(float8_e4m3&& o) = default;
  ~float8_e4m3() = default;

  HOSTDEVICE inline explicit float8_e4m3(float val) {
#if defined(PADDLE_CUDA_FP8)
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(val);
    x = *reinterpret_cast<uint8_t*>(&tmp);
#else
    PADDLE_THROW(
        phi::errors::Unimplemented("float8 is cannot be converted from float "
                                   "if __nv_fp8_e4m3 is not used"));
#endif
  }

#if defined(PADDLE_CUDA_FP8)
  HOSTDEVICE inline explicit float8_e4m3(const __nv_fp8_e4m3& val) {
    x = *reinterpret_cast<const uint8_t*>(&val);  // NOLINT
  }
#endif

  template <class T>
  HOSTDEVICE inline explicit float8_e4m3(const T& val)
      : x(float8_e4m3(static_cast<float>(val)).x) {}

// Assignment operators
#if defined(PADDLE_CUDA_FP8)
  HOSTDEVICE inline float8_e4m3& operator=(const __nv_fp8_e4m3& val) {
    x = *reinterpret_cast<const uint8_t*>(&val);  // NOLINT
    return *this;
  }
#endif

  HOSTDEVICE inline float8_e4m3& operator=(int8_t val) {
    x = float8_e4m3(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3& operator=(uint8_t val) {
    x = float8_e4m3(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3& operator=(int16_t val) {
    x = float8_e4m3(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3& operator=(uint16_t val) {
    x = float8_e4m3(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3& operator=(int32_t val) {
    x = float8_e4m3(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3& operator=(uint32_t val) {
    x = float8_e4m3(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3& operator=(int64_t val) {
    x = float8_e4m3(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3& operator=(uint64_t val) {
    x = float8_e4m3(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3& operator=(float val) {
    x = float8_e4m3(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3& operator=(double val) {
    x = float8_e4m3(val).x;
    return *this;
  }

  // Conversion operators
  HOSTDEVICE inline operator float() const {
#ifdef PADDLE_CUDA_FP8
    return float(*reinterpret_cast<const __nv_fp8_e4m3*>(&x));  // NOLINT
#else
    PADDLE_THROW(phi::errors::Unimplemented(
        "float8 is cannot be converted to float if __nv_fp8_e4m3 is not used"));
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
};

HOSTDEVICE inline float8_e4m3 operator+(const float8_e4m3& a,
                                        const float8_e4m3& b) {
  return float8_e4m3(static_cast<float>(a) + static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3 operator-(const float8_e4m3& a,
                                        const float8_e4m3& b) {
  return float8_e4m3(static_cast<float>(a) - static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3 operator*(const float8_e4m3& a,
                                        const float8_e4m3& b) {
  return float8_e4m3(static_cast<float>(a) * static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3 operator/(const float8_e4m3& a,
                                        const float8_e4m3& b) {
  return float8_e4m3(static_cast<float>(a) / static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3 operator-(const float8_e4m3& a) {
  return float8_e4m3(-static_cast<float>(a));
}

HOSTDEVICE inline float8_e4m3& operator+=(float8_e4m3& a,  // NOLINT
                                          const float8_e4m3& b) {
  a = float8_e4m3(static_cast<float>(a) + static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3& operator-=(float8_e4m3& a,  // NOLINT
                                          const float8_e4m3& b) {
  a = float8_e4m3(static_cast<float>(a) - static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3& operator*=(float8_e4m3& a,  // NOLINT
                                          const float8_e4m3& b) {
  a = float8_e4m3(static_cast<float>(a) * static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3& operator/=(float8_e4m3& a,  // NOLINT
                                          const float8_e4m3& b) {
  a = float8_e4m3(static_cast<float>(a) / static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3 raw_uint8_to_float8_e4m3(uint8_t a) {
  float8_e4m3 res;
  res.x = a;
  return res;
}

// Comparison operators
HOSTDEVICE inline bool operator==(const float8_e4m3& a, const float8_e4m3& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

HOSTDEVICE inline bool operator!=(const float8_e4m3& a, const float8_e4m3& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

HOSTDEVICE inline bool operator<(const float8_e4m3& a, const float8_e4m3& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

HOSTDEVICE inline bool operator<=(const float8_e4m3& a, const float8_e4m3& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

HOSTDEVICE inline bool operator>(const float8_e4m3& a, const float8_e4m3& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

HOSTDEVICE inline bool operator>=(const float8_e4m3& a, const float8_e4m3& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

HOSTDEVICE inline bool(isnan)(const float8_e4m3& a) {
  return (a.x & 0b01111111) == 0b01111111;
}

// TODO(Wanglongzhi2001): According to https://arxiv.org/pdf/2209.05433.pdf,
// inf is undefined in float8_e4m3, so we use 0b01111000 to represent inf for
// now!
HOSTDEVICE inline bool(isinf)(const float8_e4m3& a) {
  return (a.x & 0b01111111) == 0b01111000;
}

HOSTDEVICE inline bool(isfinite)(const float8_e4m3& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

HOSTDEVICE inline float8_e4m3(abs)(const float8_e4m3& a) {
  return float8_e4m3(std::abs(static_cast<float>(a)));
}

inline std::ostream& operator<<(std::ostream& os, const float8_e4m3& a) {
  os << static_cast<float>(a);
  return os;
}

}  // namespace dtype
}  // namespace phi

namespace std {

template <>
struct is_pod<phi::dtype::float8_e4m3> {
  static const bool value = is_trivial<phi::dtype::float8_e4m3>::value &&
                            is_standard_layout<phi::dtype::float8_e4m3>::value;
};

template <>
struct is_floating_point<phi::dtype::float8_e4m3>
    : std::integral_constant<
          bool,
          std::is_same<
              phi::dtype::float8_e4m3,
              typename std::remove_cv<phi::dtype::float8_e4m3>::type>::value> {
};
template <>
struct is_signed<phi::dtype::float8_e4m3> {
  static const bool value = true;
};

template <>
struct is_unsigned<phi::dtype::float8_e4m3> {
  static const bool value = false;
};

inline bool isnan(const phi::dtype::float8_e4m3& a) {
  return phi::dtype::isnan(a);
}

inline bool isinf(const phi::dtype::float8_e4m3& a) {
  return phi::dtype::isinf(a);
}

template <>
struct numeric_limits<phi::dtype::float8_e4m3> {
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
  static constexpr int min_exponent = -6;
  static constexpr int min_exponent10 = -1;
  static constexpr int max_exponent = 8;
  static constexpr int max_exponent10 = 2;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = false;

  HOSTDEVICE static phi::dtype::float8_e4m3(min)() {
    return phi::dtype::raw_uint8_to_float8_e4m3(0x08);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3 lowest() {
    return phi::dtype::raw_uint8_to_float8_e4m3(0xFF);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3(max)() {
    return phi::dtype::raw_uint8_to_float8_e4m3(0x7F);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3 epsilon() {
    return phi::dtype::raw_uint8_to_float8_e4m3(0x28);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3 round_error() {
    return phi::dtype::raw_uint8_to_float8_e4m3(0x38);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3 infinity() {
    return phi::dtype::raw_uint8_to_float8_e4m3(0x80);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3 quiet_NaN() {
    return phi::dtype::raw_uint8_to_float8_e4m3(0x80);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3 denorm_min() {
    return phi::dtype::raw_uint8_to_float8_e4m3(0x01);
  }
};

}  // namespace std
