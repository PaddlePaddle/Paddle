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
#include "paddle/common/data_type_util.h"
#include "paddle/common/hostdevice.h"

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

struct PADDLE_ALIGN(1) float8_e4m3fnuz {
 public:
  uint8_t x;

  // Constructors
  float8_e4m3fnuz() = default;
  float8_e4m3fnuz(const float8_e4m3fnuz& o) = default;
  float8_e4m3fnuz& operator=(const float8_e4m3fnuz& o) = default;
  float8_e4m3fnuz(float8_e4m3fnuz&& o) = default;
  float8_e4m3fnuz& operator=(float8_e4m3fnuz&& o) = default;
  ~float8_e4m3fnuz() = default;

  HOSTDEVICE inline float8_e4m3fnuz(float val) {
    x = fp8e4m3fnuz_from_fp32_value(val);
  }

  template <class T>
  HOSTDEVICE inline explicit float8_e4m3fnuz(const T& val)
      : x(float8_e4m3fnuz(static_cast<float>(val)).x) {}

  HOSTDEVICE inline float8_e4m3fnuz& operator=(int8_t val) {
    x = float8_e4m3fnuz(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fnuz& operator=(uint8_t val) {
    x = float8_e4m3fnuz(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fnuz& operator=(int16_t val) {
    x = float8_e4m3fnuz(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fnuz& operator=(uint16_t val) {
    x = float8_e4m3fnuz(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fnuz& operator=(int32_t val) {
    x = float8_e4m3fnuz(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fnuz& operator=(uint32_t val) {
    x = float8_e4m3fnuz(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fnuz& operator=(int64_t val) {
    x = float8_e4m3fnuz(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fnuz& operator=(uint64_t val) {
    x = float8_e4m3fnuz(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fnuz& operator=(float val) {
    x = float8_e4m3fnuz(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e4m3fnuz& operator=(double val) {
    x = float8_e4m3fnuz(val).x;
    return *this;
  }

  // Conversion operators
  HOSTDEVICE inline operator float() const {
    return fp8_fnuz_to_fp32_value<4, 3>(x);
  }

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
// Vector types
struct PADDLE_ALIGN(2) float8e4m3fnuz2 {
  float8_e4m3fnuz x, y;
};
struct PADDLE_ALIGN(4) float8e4m3fnuz4 {
  float8_e4m3fnuz x, y, z, w;
};

HOSTDEVICE inline float8_e4m3fnuz operator+(const float8_e4m3fnuz& a,
                                            const float8_e4m3fnuz& b) {
  return float8_e4m3fnuz(static_cast<float>(a) + static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3fnuz operator-(const float8_e4m3fnuz& a,
                                            const float8_e4m3fnuz& b) {
  return float8_e4m3fnuz(static_cast<float>(a) - static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3fnuz operator*(const float8_e4m3fnuz& a,
                                            const float8_e4m3fnuz& b) {
  return float8_e4m3fnuz(static_cast<float>(a) * static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3fnuz operator/(const float8_e4m3fnuz& a,
                                            const float8_e4m3fnuz& b) {
  return float8_e4m3fnuz(static_cast<float>(a) / static_cast<float>(b));
}

HOSTDEVICE inline float8_e4m3fnuz operator-(const float8_e4m3fnuz& a) {
  return float8_e4m3fnuz(-static_cast<float>(a));
}

HOSTDEVICE inline float8_e4m3fnuz& operator+=(float8_e4m3fnuz& a,  // NOLINT
                                              const float8_e4m3fnuz& b) {
  a = float8_e4m3fnuz(static_cast<float>(a) + static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3fnuz& operator-=(float8_e4m3fnuz& a,  // NOLINT
                                              const float8_e4m3fnuz& b) {
  a = float8_e4m3fnuz(static_cast<float>(a) - static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3fnuz& operator*=(float8_e4m3fnuz& a,  // NOLINT
                                              const float8_e4m3fnuz& b) {
  a = float8_e4m3fnuz(static_cast<float>(a) * static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3fnuz& operator/=(float8_e4m3fnuz& a,  // NOLINT
                                              const float8_e4m3fnuz& b) {
  a = float8_e4m3fnuz(static_cast<float>(a) / static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e4m3fnuz raw_uint8_to_float8_e4m3fnuz(uint8_t a) {
  float8_e4m3fnuz res;
  res.x = a;
  return res;
}

// Comparison operators
HOSTDEVICE inline bool operator==(const float8_e4m3fnuz& a,
                                  const float8_e4m3fnuz& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

HOSTDEVICE inline bool operator!=(const float8_e4m3fnuz& a,
                                  const float8_e4m3fnuz& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

HOSTDEVICE inline bool operator<(const float8_e4m3fnuz& a,
                                 const float8_e4m3fnuz& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

HOSTDEVICE inline bool operator<=(const float8_e4m3fnuz& a,
                                  const float8_e4m3fnuz& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

HOSTDEVICE inline bool operator>(const float8_e4m3fnuz& a,
                                 const float8_e4m3fnuz& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

HOSTDEVICE inline bool operator>=(const float8_e4m3fnuz& a,
                                  const float8_e4m3fnuz& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

HOSTDEVICE inline bool(isnan)(const float8_e4m3fnuz& a) {
  return a.x == 0b10000000;
}

HOSTDEVICE inline float8_e4m3fnuz(abs)(const float8_e4m3fnuz& a) {
  return float8_e4m3fnuz(std::abs(static_cast<float>(a)));
}

inline std::ostream& operator<<(std::ostream& os, const float8_e4m3fnuz& a) {
  os << static_cast<float>(a);
  return os;
}

}  // namespace dtype
}  // namespace phi

namespace cinn {
namespace common {
using float8_e4m3fnuz = ::phi::dtype::float8_e4m3fnuz;
}  // namespace common
}  // namespace cinn

namespace std {

template <>
struct is_pod<phi::dtype::float8_e4m3fnuz> {
  static const bool value =
      is_trivial<phi::dtype::float8_e4m3fnuz>::value &&
      is_standard_layout<phi::dtype::float8_e4m3fnuz>::value;
};

template <>
struct is_floating_point<phi::dtype::float8_e4m3fnuz>
    : std::integral_constant<
          bool,
          std::is_same<phi::dtype::float8_e4m3fnuz,
                       typename std::remove_cv<
                           phi::dtype::float8_e4m3fnuz>::type>::value> {};
template <>
struct is_signed<phi::dtype::float8_e4m3fnuz> {
  static const bool value = true;
};

template <>
struct is_unsigned<phi::dtype::float8_e4m3fnuz> {
  static const bool value = false;
};

inline bool isnan(const phi::dtype::float8_e4m3fnuz& a) {
  return phi::dtype::isnan(a);
}

template <>
struct numeric_limits<phi::dtype::float8_e4m3fnuz> {
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

  HOSTDEVICE static phi::dtype::float8_e4m3fnuz(min)() {
    return phi::dtype::raw_uint8_to_float8_e4m3fnuz(0x08);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fnuz lowest() {
    return phi::dtype::raw_uint8_to_float8_e4m3fnuz(0xFF);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fnuz(max)() {
    return phi::dtype::raw_uint8_to_float8_e4m3fnuz(0x7F);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fnuz epsilon() {
    return phi::dtype::raw_uint8_to_float8_e4m3fnuz(0x28);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fnuz round_error() {
    return phi::dtype::raw_uint8_to_float8_e4m3fnuz(0x38);
  }

  HOSTDEVICE static phi::dtype::float8_e4m3fnuz infinity() {
    return phi::dtype::raw_uint8_to_float8_e4m3fnuz(0x80);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fnuz quiet_NaN() {
    return phi::dtype::raw_uint8_to_float8_e4m3fnuz(0x80);
  }
  HOSTDEVICE static phi::dtype::float8_e4m3fnuz denorm_min() {
    return phi::dtype::raw_uint8_to_float8_e4m3fnuz(0x01);
  }
};
template <>
struct common_type<float, phi::dtype::float8_e4m3fnuz> {
  using type = float;
};
}  // namespace std
