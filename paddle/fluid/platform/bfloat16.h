// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <limits>
#if !defined(_WIN32)
#define PADDLE_ALIGN(x) __attribute__((aligned(x)))
#else
#define PADDLE_ALIGN(x) __declspec(align(x))
#endif

#include <cstring>

#include "paddle/fluid/platform/hostdevice.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace Eigen {
template <typename T>
struct NumTraits;
}  // namespace Eigen

namespace paddle {
namespace platform {

struct PADDLE_ALIGN(2) bfloat16 {
 public:
  uint16_t x;

  bfloat16() = default;
  bfloat16(const bfloat16& o) = default;
  bfloat16& operator=(const bfloat16& o) = default;
  bfloat16(bfloat16&& o) = default;
  bfloat16& operator=(bfloat16&& o) = default;
  ~bfloat16() = default;

  HOSTDEVICE inline explicit bfloat16(float val) {
    std::memcpy(&x, reinterpret_cast<char*>(&val) + 2, 2);
  }

  template <class T>
  HOSTDEVICE inline explicit bfloat16(const T& val)
      : x(bfloat16(static_cast<float>(val)).x) {}

  HOSTDEVICE inline bfloat16& operator=(bool b) {
    x = b ? 0x3f80 : 0;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(int64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(uint64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(float val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline bfloat16& operator=(double val) {
    x = bfloat16(val).x;
    return *this;
  }

  HOSTDEVICE inline explicit operator float() const {
    float val = 0.f;
    uint16_t temp = x;
    memcpy(reinterpret_cast<char*>(&val) + 2, reinterpret_cast<char*>(&temp),
           2);
    return val;
  }

  HOSTDEVICE inline explicit operator bool() const { return (x & 0x7fff) != 0; }

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

  HOSTDEVICE inline explicit operator double() const {
    return static_cast<double>(static_cast<float>(*this));
  }
};

HOSTDEVICE inline bfloat16 operator+(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}

HOSTDEVICE inline bfloat16 operator-(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) - static_cast<float>(b));
}

HOSTDEVICE inline bfloat16 operator*(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) * static_cast<float>(b));
}

HOSTDEVICE inline bfloat16 operator/(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) / static_cast<float>(b));
}

HOSTDEVICE inline bfloat16 operator-(const bfloat16& a) {
  bfloat16 res;
  res.x = a.x ^ 0x8000;
  return res;
}

HOSTDEVICE inline bfloat16& operator+=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) + static_cast<float>(b));
  return a;
}

HOSTDEVICE inline bfloat16& operator-=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) - static_cast<float>(b));
  return a;
}

HOSTDEVICE inline bfloat16& operator*=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) * static_cast<float>(b));
  return a;
}

HOSTDEVICE inline bfloat16& operator/=(bfloat16& a,  // NOLINT
                                       const bfloat16& b) {
  a = bfloat16(static_cast<float>(a) / static_cast<float>(b));
  return a;
}

HOSTDEVICE inline bfloat16 raw_uint16_to_bfloat16(uint16_t a) {
  bfloat16 res;
  res.x = a;
  return res;
}

HOSTDEVICE inline bool operator==(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

HOSTDEVICE inline bool operator!=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

HOSTDEVICE inline bool operator<(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

HOSTDEVICE inline bool operator<=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

HOSTDEVICE inline bool operator>(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

HOSTDEVICE inline bool operator>=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

HOSTDEVICE inline bool(isnan)(const bfloat16& a) {
  return (a.x & 0x7FFF) > 0x7F80;
}

HOSTDEVICE inline bool(isinf)(const bfloat16& a) {
  return (a.x & 0x7F80) == 0x7F80;
}

HOSTDEVICE inline bool(isfinite)(const bfloat16& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

inline std::ostream& operator<<(std::ostream& os, const bfloat16& a) {
  os << a.x;
  return os;
}

}  // namespace platform
}  // namespace paddle

namespace std {

template <>
struct is_pod<paddle::platform::bfloat16> {
  static const bool value =
      is_trivial<paddle::platform::bfloat16>::value &&
      is_standard_layout<paddle::platform::bfloat16>::value;
};

template <>
struct is_floating_point<paddle::platform::bfloat16>
    : std::integral_constant<
          bool, std::is_same<paddle::platform::bfloat16,
                             typename std::remove_cv<
                                 paddle::platform::bfloat16>::type>::value> {};
template <>
struct is_signed<paddle::platform::bfloat16> {
  static const bool value = true;
};

template <>
struct is_unsigned<paddle::platform::bfloat16> {
  static const bool value = false;
};

inline bool isnan(const paddle::platform::bfloat16& a) {
  return paddle::platform::isnan(a);
}

inline bool isinf(const paddle::platform::bfloat16& a) {
  return paddle::platform::isinf(a);
}

template <>
struct numeric_limits<paddle::platform::bfloat16> {
  static const bool is_specialized = true;
  static const bool is_signed = true;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const bool has_infinity = true;
  static const bool has_quiet_NaN = true;
  static const bool has_signaling_NaN = true;
  static const float_denorm_style has_denorm = denorm_present;
  static const bool has_denorm_loss = false;
  static const std::float_round_style round_style = std::round_to_nearest;
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  static const int digits = 8;
  static const int digits10 = 2;
  static const int max_digits10 = 9;
  static const int radix = 2;
  static const int min_exponent = -125;
  static const int min_exponent10 = -37;
  static const int max_exponent = 128;
  static const int max_exponent10 = 38;
  static const bool traps = true;
  static const bool tinyness_before = false;

  static paddle::platform::bfloat16(min)() {
    return paddle::platform::raw_uint16_to_bfloat16(0x007f);
  }
  static paddle::platform::bfloat16 lowest() {
    return paddle::platform::raw_uint16_to_bfloat16(0xff7f);
  }
  static paddle::platform::bfloat16(max)() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7f7f);
  }
  static paddle::platform::bfloat16 epsilon() {
    return paddle::platform::raw_uint16_to_bfloat16(0x3400);
  }
  static paddle::platform::bfloat16 round_error() {
    return paddle::platform::bfloat16(0.5);
  }
  static paddle::platform::bfloat16 infinity() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7f80);
  }
  static paddle::platform::bfloat16 quiet_NaN() {
    return paddle::platform::raw_uint16_to_bfloat16(0xffc1);
  }
  static paddle::platform::bfloat16 signaling_NaN() {
    return paddle::platform::raw_uint16_to_bfloat16(0xff81);
  }
  static paddle::platform::bfloat16 denorm_min() {
    return paddle::platform::raw_uint16_to_bfloat16(0x0001);
  }
};

}  // namespace std

namespace Eigen {

using bfloat16 = paddle::platform::bfloat16;

template <>
struct NumTraits<bfloat16> : GenericNumTraits<bfloat16> {
  enum {
    IsSigned = true,
    IsInteger = false,
    IsComplex = false,
    RequireInitialization = false
  };

  HOSTDEVICE static inline bfloat16 epsilon() {
    return paddle::platform::raw_uint16_to_bfloat16(0x3400);
  }
  HOSTDEVICE static inline bfloat16 dummy_precision() {
    return bfloat16(1e-5f);
  }
  HOSTDEVICE static inline bfloat16 highest() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7f7f);
  }
  HOSTDEVICE static inline bfloat16 lowest() {
    return paddle::platform::raw_uint16_to_bfloat16(0xff7f);
  }
  HOSTDEVICE static inline bfloat16 infinity() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7f80);
  }
  HOSTDEVICE static inline bfloat16 quiet_NaN() {
    return paddle::platform::raw_uint16_to_bfloat16(0xffc1);
  }
};
namespace numext {

template <>
HOSTDEVICE inline bool(isnan)(const bfloat16& a) {
  return (paddle::platform::isnan)(a);
}

template <>
HOSTDEVICE inline bool(isinf)(const bfloat16& a) {
  return (paddle::platform::isinf)(a);
}

template <>
HOSTDEVICE inline bool(isfinite)(const bfloat16& a) {
  return (paddle::platform::isfinite)(a);
}

template <>
HOSTDEVICE inline bfloat16 exp(const bfloat16& a) {
  return bfloat16(::expf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline bfloat16 erf(const bfloat16& a) {
  return bfloat16(::erff(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline bfloat16 log(const bfloat16& a) {
  return bfloat16(::logf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline bfloat16 tanh(const bfloat16& a) {
  return bfloat16(::tanhf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline bfloat16 sqrt(const bfloat16& a) {
  return bfloat16(::sqrtf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline bfloat16 ceil(const bfloat16& a) {
  return bfloat16(::ceilf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline bfloat16 floor(const bfloat16& a) {
  return bfloat16(::floorf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline bfloat16 round(const bfloat16& a) {
  return bfloat16(::roundf(static_cast<float>(a)));
}

template <>
HOSTDEVICE inline bfloat16 pow(const bfloat16& a, const bfloat16& b) {
  return bfloat16(::powf(static_cast<float>(a), static_cast<float>(b)));
}

template <>
HOSTDEVICE inline bfloat16 abs(const bfloat16& a) {
  return bfloat16(::fabs(static_cast<float>(a)));
}

}  // namespace numext
}  // namespace Eigen
