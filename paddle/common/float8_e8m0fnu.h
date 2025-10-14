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

struct PADDLE_ALIGN(1) float8_e8m0fnu {
 public:
  uint8_t x;

  // Constructors
  float8_e8m0fnu() = default;
  float8_e8m0fnu(const float8_e8m0fnu& o) = default;
  float8_e8m0fnu& operator=(const float8_e8m0fnu& o) = default;
  float8_e8m0fnu(float8_e8m0fnu&& o) = default;
  float8_e8m0fnu& operator=(float8_e8m0fnu&& o) = default;
  ~float8_e8m0fnu() = default;

  HOSTDEVICE inline float8_e8m0fnu(float val) {
    x = fp8e8m0fnu_from_fp32_value(val);
  }

  template <class T>
  HOSTDEVICE inline explicit float8_e8m0fnu(const T& val)
      : x(float8_e8m0fnu(static_cast<float>(val)).x) {}

  HOSTDEVICE inline float8_e8m0fnu& operator=(int8_t val) {
    x = float8_e8m0fnu(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e8m0fnu& operator=(uint8_t val) {
    x = float8_e8m0fnu(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e8m0fnu& operator=(int16_t val) {
    x = float8_e8m0fnu(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e8m0fnu& operator=(uint16_t val) {
    x = float8_e8m0fnu(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e8m0fnu& operator=(int32_t val) {
    x = float8_e8m0fnu(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e8m0fnu& operator=(uint32_t val) {
    x = float8_e8m0fnu(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e8m0fnu& operator=(int64_t val) {
    x = float8_e8m0fnu(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e8m0fnu& operator=(uint64_t val) {
    x = float8_e8m0fnu(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e8m0fnu& operator=(float val) {
    x = float8_e8m0fnu(val).x;
    return *this;
  }

  HOSTDEVICE inline float8_e8m0fnu& operator=(double val) {
    x = float8_e8m0fnu(val).x;
    return *this;
  }

  // Conversion operators
  HOSTDEVICE inline operator float() const {
    Bits fb;
    if (x == 0) {
      fb.ui = 0x00400000;
      return fb.f;
    } else if (x == 0b11111111) {
      fb.ui = 0x7f800001;
      return fb.f;
    }
    fb.ui = x << 23;
    return fb.f;
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

HOSTDEVICE inline float8_e8m0fnu operator+(const float8_e8m0fnu& a,
                                           const float8_e8m0fnu& b) {
  return float8_e8m0fnu(static_cast<float>(a) + static_cast<float>(b));
}

HOSTDEVICE inline float8_e8m0fnu operator-(const float8_e8m0fnu& a,
                                           const float8_e8m0fnu& b) {
  return float8_e8m0fnu(static_cast<float>(a) - static_cast<float>(b));
}

HOSTDEVICE inline float8_e8m0fnu operator*(const float8_e8m0fnu& a,
                                           const float8_e8m0fnu& b) {
  return float8_e8m0fnu(static_cast<float>(a) * static_cast<float>(b));
}

HOSTDEVICE inline float8_e8m0fnu operator/(const float8_e8m0fnu& a,
                                           const float8_e8m0fnu& b) {
  return float8_e8m0fnu(static_cast<float>(a) / static_cast<float>(b));
}

HOSTDEVICE inline float8_e8m0fnu operator-(const float8_e8m0fnu& a) {
  return float8_e8m0fnu(-static_cast<float>(a));
}

HOSTDEVICE inline float8_e8m0fnu& operator+=(float8_e8m0fnu& a,  // NOLINT
                                             const float8_e8m0fnu& b) {
  a = float8_e8m0fnu(static_cast<float>(a) + static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e8m0fnu& operator-=(float8_e8m0fnu& a,  // NOLINT
                                             const float8_e8m0fnu& b) {
  a = float8_e8m0fnu(static_cast<float>(a) - static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e8m0fnu& operator*=(float8_e8m0fnu& a,  // NOLINT
                                             const float8_e8m0fnu& b) {
  a = float8_e8m0fnu(static_cast<float>(a) * static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e8m0fnu& operator/=(float8_e8m0fnu& a,  // NOLINT
                                             const float8_e8m0fnu& b) {
  a = float8_e8m0fnu(static_cast<float>(a) / static_cast<float>(b));
  return a;
}

HOSTDEVICE inline float8_e8m0fnu raw_uint8_to_float8_e8m0fnu(uint8_t a) {
  float8_e8m0fnu res;
  res.x = a;
  return res;
}

// Comparison operators
HOSTDEVICE inline bool operator==(const float8_e8m0fnu& a,
                                  const float8_e8m0fnu& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

HOSTDEVICE inline bool operator!=(const float8_e8m0fnu& a,
                                  const float8_e8m0fnu& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

HOSTDEVICE inline bool operator<(const float8_e8m0fnu& a,
                                 const float8_e8m0fnu& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

HOSTDEVICE inline bool operator<=(const float8_e8m0fnu& a,
                                  const float8_e8m0fnu& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

HOSTDEVICE inline bool operator>(const float8_e8m0fnu& a,
                                 const float8_e8m0fnu& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

HOSTDEVICE inline bool operator>=(const float8_e8m0fnu& a,
                                  const float8_e8m0fnu& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

HOSTDEVICE inline bool(isnan)(const float8_e8m0fnu& a) {
  return a.x == 0b11111111;
}

HOSTDEVICE inline float8_e8m0fnu(abs)(const float8_e8m0fnu& a) {
  return float8_e8m0fnu(std::abs(static_cast<float>(a)));
}

inline std::ostream& operator<<(std::ostream& os, const float8_e8m0fnu& a) {
  os << static_cast<float>(a);
  return os;
}

}  // namespace dtype
}  // namespace phi

namespace cinn {
namespace common {
using float8_e8m0fnu = ::phi::dtype::float8_e8m0fnu;
}  // namespace common
}  // namespace cinn

namespace std {

template <>
struct is_pod<phi::dtype::float8_e8m0fnu> {
  static const bool value =
      is_trivial<phi::dtype::float8_e8m0fnu>::value &&
      is_standard_layout<phi::dtype::float8_e8m0fnu>::value;
};

template <>
struct is_floating_point<phi::dtype::float8_e8m0fnu>
    : std::integral_constant<
          bool,
          std::is_same<phi::dtype::float8_e8m0fnu,
                       typename std::remove_cv<
                           phi::dtype::float8_e8m0fnu>::type>::value> {};
template <>
struct is_signed<phi::dtype::float8_e8m0fnu> {
  static const bool value = true;
};

template <>
struct is_unsigned<phi::dtype::float8_e8m0fnu> {
  static const bool value = false;
};

inline bool isnan(const phi::dtype::float8_e8m0fnu& a) {
  return phi::dtype::isnan(a);
}

template <>
struct numeric_limits<phi::dtype::float8_e8m0fnu> {
  static constexpr bool is_signed = false;
  static constexpr bool is_integer = false;
  static constexpr bool is_specialized = true;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = false;
  static constexpr auto has_denorm_loss = false;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 1;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 1;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -126;
  static constexpr int min_exponent10 = -38;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = false;

  HOSTDEVICE static phi::dtype::float8_e8m0fnu(min)() {
    return phi::dtype::raw_uint8_to_float8_e8m0fnu(0x0);
  }
  HOSTDEVICE static phi::dtype::float8_e8m0fnu lowest() {
    return phi::dtype::raw_uint8_to_float8_e8m0fnu(0x0);
  }
  HOSTDEVICE static phi::dtype::float8_e8m0fnu(max)() {
    return phi::dtype::raw_uint8_to_float8_e8m0fnu(0xFE);
  }
  HOSTDEVICE static phi::dtype::float8_e8m0fnu epsilon() {
    return phi::dtype::raw_uint8_to_float8_e8m0fnu(0x7F);
  }
  HOSTDEVICE static phi::dtype::float8_e8m0fnu round_error() {
    return phi::dtype::raw_uint8_to_float8_e8m0fnu(0x7E);
  }
  HOSTDEVICE static phi::dtype::float8_e8m0fnu infinity() {
    return phi::dtype::raw_uint8_to_float8_e8m0fnu(0xFF);
  }  // NaN.
  HOSTDEVICE static phi::dtype::float8_e8m0fnu quiet_NaN() {
    return phi::dtype::raw_uint8_to_float8_e8m0fnu(0xFF);
  }
};

}  // namespace std
