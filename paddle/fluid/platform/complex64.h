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

struct PADDLE_ALIGN(8) complex64 {
 public:
  float real;
  float imag;

  complex64() = default;
  complex64(float real, float imag): real(real), imag(imag) {}
  complex64(const complex64& o) = default;
  complex64& operator=(const complex64& o) = default;
  complex64(complex64&& o) = default;
  complex64& operator=(complex64&& o) = default;
  ~complex64() = default;

  HOSTDEVICE inline explicit complex64(float val) {
    std::memcpy(&real, &val, 4);
  }

  template <class T>
  HOSTDEVICE inline explicit complex64(const T& val)
      : real(complex64(static_cast<float>(val)).real) {}

  HOSTDEVICE inline complex64& operator=(bool b) {
    real = b ? 1 : 0;
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(int8_t val) {
    real = float(val);
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(uint8_t val) {
    real = float(val);
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(int16_t val) {
    real = float(val);
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(uint16_t val) {
    real = float(val);
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(int32_t val) {
    real = float(val);
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(uint32_t val) {
    real = float(val);
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(int64_t val) {
    real = float(val);
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(uint64_t val) {
    real = float(val);
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(float val) {
    real = val;
    return *this;
  }

  HOSTDEVICE inline complex64& operator=(double val) {
    real = float(val);
    return *this;
  }

  HOSTDEVICE inline explicit operator float() const {
    return this->real;
  }

  HOSTDEVICE inline explicit operator bool() const { 
    return bool(this->real) || bool(this->imag); 
  }

  HOSTDEVICE inline explicit operator int8_t() const {
    return static_cast<int8_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int16_t() const {
    return static_cast<int16_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int32_t() const {
    return static_cast<int32_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int64_t() const {
    return static_cast<int64_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(this->real);
  }

  HOSTDEVICE inline explicit operator double() const {
    return static_cast<double>(this->real);
  }
};

HOSTDEVICE inline complex64 operator+(const complex64& a, const complex64& b) {
  //return complex64(a.real + b.real, a.imag + b.imag);
  return complex64();
}

HOSTDEVICE inline complex64 operator-(const complex64& a, const complex64& b) {
  return complex64(a.real - b.real, a.imag - b.imag);
}

HOSTDEVICE inline complex64 operator*(const complex64& a, const complex64& b) {
  return complex64(a.real * b.real - a.imag * b.imag, a.imag * b.real + b.imag * a.real);
}

HOSTDEVICE inline complex64 operator/(const complex64& a, const complex64& b) {
  float denominator = b.real * b.real + b.imag * b.imag;
  return complex64((a.real * b.real + a.imag * b.imag)/ denominator, (a.imag * b.real - a.real * b.imag)/ denominator);
}

HOSTDEVICE inline complex64 operator-(const complex64& a) {
  complex64 res;
  res.real = 0 - a.real;
  return res;
}

HOSTDEVICE inline complex64& operator+=(complex64& a,  // NOLINT
                                       const complex64& b) {
  a = complex64(a.real + b.real, a.imag + b.imag);
  return a;
}

HOSTDEVICE inline complex64& operator-=(complex64& a,  // NOLINT
                                       const complex64& b) {
  a = complex64(a.real - b.real, a.imag - b.imag);
  return a;
}

HOSTDEVICE inline complex64& operator*=(complex64& a,  // NOLINT
                                       const complex64& b) {
  a = complex64(a.real * b.real - a.imag * b.imag, a.imag * b.real + b.imag * a.real);
  return a;
}

HOSTDEVICE inline complex64& operator/=(complex64& a,  // NOLINT
                                       const complex64& b) {
  float denominator = b.real * b.real + b.imag * b.imag;
  a = complex64((a.real * b.real + a.imag * b.imag)/ denominator, (a.imag * b.real - a.real * b.imag)/ denominator);
  return a;
}
HOSTDEVICE inline complex64 raw_uint16_to_complex64(uint16_t a) {
  complex64 res;
  res.real = a;
  return res;
}

HOSTDEVICE inline bool operator==(const complex64& a, const complex64& b) {
  return a.real == b.real && a.imag == b.imag;
}

HOSTDEVICE inline bool operator!=(const complex64& a, const complex64& b) {
  return a.real != b.real || a.imag != b.imag;
}

HOSTDEVICE inline bool operator<(const complex64& a, const complex64& b) {
  return static_cast<float>(a.real) < static_cast<float>(b.real);
}

HOSTDEVICE inline bool operator<=(const complex64& a, const complex64& b) {
  return static_cast<float>(a.real) <= static_cast<float>(b.real);
}

HOSTDEVICE inline bool operator>(const complex64& a, const complex64& b) {
  return static_cast<float>(a.real) > static_cast<float>(b.real);
}

HOSTDEVICE inline bool operator>=(const complex64& a, const complex64& b) {
  return static_cast<float>(a.real) >= static_cast<float>(b.real);
}

HOSTDEVICE inline bool(isnan)(const complex64& a) {
  return std::isnan(a.real) || std::isnan(a.imag);
}

HOSTDEVICE inline bool(isinf)(const complex64& a) {
  return std::isinf(a.real) || std::isinf(a.imag);
}

HOSTDEVICE inline bool(isfinite)(const complex64& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

inline std::ostream& operator<<(std::ostream& os, const complex64& a) {
  os << "real:" << a.real << " imag:" << a.imag;
  return os;
}

}  // namespace platform
}  // namespace paddle

namespace std {

template <>
struct is_pod<paddle::platform::complex64> {
  static const bool value =
      is_trivial<paddle::platform::complex64>::value &&
      is_standard_layout<paddle::platform::complex64>::value;
};

template <>
struct is_floating_point<paddle::platform::complex64>
    : std::integral_constant<
          bool, std::is_same<paddle::platform::complex64,
                             typename std::remove_cv<
                                 paddle::platform::complex64>::type>::value> {};
template <>
struct is_signed<paddle::platform::complex64> {
  static const bool value = true;
};

template <>
struct is_unsigned<paddle::platform::complex64> {
  static const bool value = false;
};

inline bool isnan(const paddle::platform::complex64& a) {
  return paddle::platform::isnan(a);
}

inline bool isinf(const paddle::platform::complex64& a) {
  return paddle::platform::isinf(a);
}

template <>
struct numeric_limits<paddle::platform::complex64> {
  static const bool is_specialized = true;
  //static const bool is_signed = true;
  static const bool is_signed = false;
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

  static paddle::platform::complex64(min)() {
    return paddle::platform::raw_uint16_to_complex64(0x007f);
  }
  static paddle::platform::complex64 lowest() {
    return paddle::platform::raw_uint16_to_complex64(0xff7f);
  }
  static paddle::platform::complex64(max)() {
    return paddle::platform::raw_uint16_to_complex64(0x7f7f);
  }
  static paddle::platform::complex64 epsilon() {
    return paddle::platform::raw_uint16_to_complex64(0x3400);
  }
  static paddle::platform::complex64 round_error() {
    return paddle::platform::complex64(0.5);
  }
  static paddle::platform::complex64 infinity() {
    return paddle::platform::raw_uint16_to_complex64(0x7f80);
  }
  static paddle::platform::complex64 quiet_NaN() {
    return paddle::platform::raw_uint16_to_complex64(0xffc1);
  }
  static paddle::platform::complex64 signaling_NaN() {
    return paddle::platform::raw_uint16_to_complex64(0xff81);
  }
  static paddle::platform::complex64 denorm_min() {
    return paddle::platform::raw_uint16_to_complex64(0x0001);
  }
};

}  // namespace std

namespace Eigen {

using complex64 = paddle::platform::complex64;

template <>
struct NumTraits<complex64> : GenericNumTraits<complex64> {
  enum {
    IsSigned = true,
    IsInteger = false,
    IsComplex = false,
    RequireInitialization = false
  };
  HOSTDEVICE static inline complex64 epsilon() {
    return paddle::platform::raw_uint16_to_complex64(0x3400);
  }
  HOSTDEVICE static inline complex64 dummy_precision() {
    return complex64(1e-5f);
  }
  HOSTDEVICE static inline complex64 highest() {
    return paddle::platform::raw_uint16_to_complex64(0x7f7f);
  }
  HOSTDEVICE static inline complex64 lowest() {
    return paddle::platform::raw_uint16_to_complex64(0xff7f);
  }
  HOSTDEVICE static inline complex64 infinity() {
    return paddle::platform::raw_uint16_to_complex64(0x7f80);
  }
  HOSTDEVICE static inline complex64 quiet_NaN() {
    return paddle::platform::raw_uint16_to_complex64(0xffc1);
  }
};
namespace numext {

template <>
HOSTDEVICE inline bool(isnan)(const complex64& a) {
  return (std::isnan)(a.real) || (std::isnan)(a.imag);
}

template <>
HOSTDEVICE inline bool(isinf)(const complex64& a) {
  return (std::isinf)(a.real) || (std::isinf)(a.imag);
}

template <>
HOSTDEVICE inline bool(isfinite)(const complex64& a) {
  return (std::isfinite)(a.real) || (std::isfinite)(a.imag);
}

// todo not impl
template <>
HOSTDEVICE inline complex64 exp(const complex64& a) {
  return complex64(::expf(static_cast<float>(a.real)));
}

template <>
HOSTDEVICE inline complex64 erf(const complex64& a) {
  return complex64(::erff(static_cast<float>(a.real)));
}

template <>
HOSTDEVICE inline complex64 log(const complex64& a) {
  return complex64(::logf(static_cast<float>(a.real)));
}

template <>
HOSTDEVICE inline complex64 tanh(const complex64& a) {
  return complex64(::tanhf(static_cast<float>(a.real)));
}

template <>
HOSTDEVICE inline complex64 sqrt(const complex64& a) {
  return complex64(::sqrtf(static_cast<float>(a.real)));
}

template <>
HOSTDEVICE inline complex64 ceil(const complex64& a) {
  return complex64(::ceilf(static_cast<float>(a.real)));
}

template <>
HOSTDEVICE inline complex64 floor(const complex64& a) {
  return complex64(::floorf(static_cast<float>(a.real)));
}

template <>
HOSTDEVICE inline complex64 round(const complex64& a) {
  return complex64(::roundf(static_cast<float>(a.real)));
}

template <>
HOSTDEVICE inline complex64 pow(const complex64& a, const complex64& b) {
  return complex64(::powf(static_cast<float>(a.real), static_cast<float>(b.real)));
}

template <>
HOSTDEVICE inline complex64 abs(const complex64& a) {
  return complex64(::fabs(static_cast<float>(a.real)));
}

}  // namespace numext
}  // namespace Eigen
