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

#ifdef PADDLE_WITH_CUDA
#include <cuComplex.h>
#include <thrust/complex.h>
#endif  // PADDLE_WITH_CUDA

#include <cstring>

#include "paddle/fluid/platform/hostdevice.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace Eigen {
template <typename T>
struct NumTraits;
}  // namespace Eigen

namespace paddle {
namespace platform {

struct PADDLE_ALIGN(16) complex128 {
 public:
  double real;
  double imag;

  complex128() = default;
  complex128(const complex128& o) = default;
  complex128& operator=(const complex128& o) = default;
  complex128(complex128&& o) = default;
  complex128& operator=(complex128&& o) = default;
  ~complex128() = default;

  HOSTDEVICE complex128(double real, double imag) : real(real), imag(imag) {}
#if defined(PADDLE_WITH_CUDA)

  HOSTDEVICE inline explicit complex128(const thrust::complex<double>& c) {
    real = c.real();
    imag = c.imag();
  }

  HOSTDEVICE inline explicit operator thrust::complex<double>() const {
    return thrust::complex<double>(real, imag);
  }

  HOSTDEVICE inline explicit operator cuDoubleComplex() const {
    return make_cuDoubleComplex(real, imag);
  }
#endif

  HOSTDEVICE complex128(const float& val) { real = static_cast<double>(val); }
  HOSTDEVICE complex128(const double& val) { real = val; }
  HOSTDEVICE complex128(const int& val) { real = static_cast<double>(val); }
  HOSTDEVICE complex128(const int64_t& val) { real = static_cast<double>(val); }

  HOSTDEVICE inline explicit operator std::complex<double>() {
    return static_cast<std::complex<double>>(std::complex<double>(real, imag));
  }

  template <class T>
  HOSTDEVICE inline explicit complex128(const T& val)
      : real(complex128(static_cast<double>(val)).real) {}

  HOSTDEVICE complex128(const std::complex<double> val)
      : real(val.real()), imag(val.imag()) {}

  HOSTDEVICE inline complex128& operator=(bool b) {
    real = b ? 1 : 0;
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex128& operator=(int8_t val) {
    real = static_cast<double>(val);
    return *this;
  }

  HOSTDEVICE inline complex128& operator=(uint8_t val) {
    real = static_cast<double>(val);
    return *this;
  }

  HOSTDEVICE inline complex128& operator=(int16_t val) {
    real = static_cast<double>(val);
    return *this;
  }

  HOSTDEVICE inline complex128& operator=(uint16_t val) {
    real = static_cast<double>(val);
    return *this;
  }

  HOSTDEVICE inline complex128& operator=(int32_t val) {
    real = static_cast<double>(val);
    return *this;
  }

  HOSTDEVICE inline complex128& operator=(uint32_t val) {
    real = static_cast<double>(val);
    return *this;
  }

  HOSTDEVICE inline complex128& operator=(int64_t val) {
    real = static_cast<double>(val);
    return *this;
  }

  HOSTDEVICE inline complex128& operator=(uint64_t val) {
    real = static_cast<double>(val);
    return *this;
  }

  HOSTDEVICE inline complex128& operator=(float val) {
    real = val;
    return *this;
  }

  HOSTDEVICE inline complex128& operator=(double val) {
    real = static_cast<double>(val);
    return *this;
  }

  HOSTDEVICE inline operator float() const {
    return static_cast<float>(this->real);
  }

  HOSTDEVICE inline explicit operator bool() const {
    return static_cast<bool>(this->real) || static_cast<bool>(this->imag);
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

HOSTDEVICE inline complex128 operator+(const complex128& a,
                                       const complex128& b) {
#if defined(__CUDA_ARCH__)
  return complex128(thrust::complex<double>(a.real, a.imag) +
                    thrust::complex<double>(b.real, b.imag));
#else
  return complex128(a.real + b.real, a.imag + b.imag);
#endif
}

HOSTDEVICE inline complex128 operator-(const complex128& a,
                                       const complex128& b) {
#if defined(__CUDA_ARCH__)
  return complex128(thrust::complex<double>(a.real, a.imag) -
                    thrust::complex<double>(b.real, b.imag));
#else
  return complex128(a.real - b.real, a.imag - b.imag);
#endif
}

HOSTDEVICE inline complex128 operator*(const complex128& a,
                                       const complex128& b) {
#if defined(__CUDA_ARCH__)
  return complex128(thrust::complex<double>(a.real, a.imag) *
                    thrust::complex<double>(b.real, b.imag));
#else
  return complex128(a.real * b.real - a.imag * b.imag,
                    a.imag * b.real + b.imag * a.real);
#endif
}

HOSTDEVICE inline complex128 operator/(const complex128& a,
                                       const complex128& b) {
#if defined(__CUDA_ARCH__)
  return complex128(thrust::complex<double>(a.real, a.imag) /
                    thrust::complex<double>(b.real, b.imag));
#else
  double denominator = b.real * b.real + b.imag * b.imag;
  return complex128((a.real * b.real + a.imag * b.imag) / denominator,
                    (a.imag * b.real - a.real * b.imag) / denominator);
#endif
}

HOSTDEVICE inline complex128 operator-(const complex128& a) {
#if defined(__CUDA_ARCH__)
  return complex128(-thrust::complex<double>(a.real, a.imag));
#else
  complex128 res;
  res.real = -a.real;
  res.imag = -a.imag;
  return res;
#endif
}

HOSTDEVICE inline complex128& operator+=(complex128& a,  // NOLINT
                                         const complex128& b) {
#if defined(__CUDA_ARCH__)
  a = complex128(thrust::complex<double>(a.real, a.imag) +=
                 thrust::complex<double>(b.real, b.imag));
  return a;
#else
  a.real += b.real;
  a.imag += b.imag;
  return a;
#endif
}

HOSTDEVICE inline complex128& operator-=(complex128& a,  // NOLINT
                                         const complex128& b) {
#if defined(__CUDA_ARCH__)
  a = complex128(thrust::complex<double>(a.real, a.imag) -=
                 thrust::complex<double>(b.real, b.imag));
  return a;
#else
  a.real -= b.real;
  a.imag -= b.imag;
  return a;
#endif
}

HOSTDEVICE inline complex128& operator*=(complex128& a,  // NOLINT
                                         const complex128& b) {
#if defined(__CUDA_ARCH__)
  a = complex128(thrust::complex<double>(a.real, a.imag) *=
                 thrust::complex<double>(b.real, b.imag));
  return a;
#else
  a.real = a.real * b.real - a.imag * b.imag;
  a.imag = a.imag * b.real + b.imag * a.real;
  return a;
#endif
}

HOSTDEVICE inline complex128& operator/=(complex128& a,  // NOLINT
                                         const complex128& b) {
#if defined(__CUDA_ARCH__)
  a = complex128(thrust::complex<double>(a.real, a.imag) /=
                 thrust::complex<double>(b.real, b.imag));
  return a;
#else
  double denominator = b.real * b.real + b.imag * b.imag;
  a.real = (a.real * b.real + a.imag * b.imag) / denominator;
  a.imag = (a.imag * b.real - a.real * b.imag) / denominator;
  return a;
#endif
}

HOSTDEVICE inline complex128 raw_uint16_to_complex128(uint16_t a) {
  complex128 res;
  res.real = a;
  return res;
}

HOSTDEVICE inline bool operator==(const complex128& a, const complex128& b) {
  return a.real == b.real && a.imag == b.imag;
}

HOSTDEVICE inline bool operator!=(const complex128& a, const complex128& b) {
  return a.real != b.real || a.imag != b.imag;
}

HOSTDEVICE inline bool operator<(const complex128& a, const complex128& b) {
  return static_cast<double>(a.real) < static_cast<double>(b.real);
}

HOSTDEVICE inline bool operator<=(const complex128& a, const complex128& b) {
  return static_cast<double>(a.real) <= static_cast<double>(b.real);
}

HOSTDEVICE inline bool operator>(const complex128& a, const complex128& b) {
  return static_cast<double>(a.real) > static_cast<double>(b.real);
}

HOSTDEVICE inline bool operator>=(const complex128& a, const complex128& b) {
  return static_cast<double>(a.real) >= static_cast<double>(b.real);
}

HOSTDEVICE inline bool(isnan)(const complex128& a) {
  return std::isnan(a.real) || std::isnan(a.imag);
}

HOSTDEVICE inline bool(isinf)(const complex128& a) {
  return std::isinf(a.real) || std::isinf(a.imag);
}

HOSTDEVICE inline bool(isfinite)(const complex128& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

inline std::ostream& operator<<(std::ostream& os, const complex128& a) {
  os << "real:" << a.real << " imag:" << a.imag;
  return os;
}

}  // namespace platform
}  // namespace paddle

namespace std {

template <>
struct is_pod<paddle::platform::complex128> {
  static const bool value =
      is_trivial<paddle::platform::complex128>::value &&
      is_standard_layout<paddle::platform::complex128>::value;
};

template <>
struct is_floating_point<paddle::platform::complex128>
    : std::integral_constant<
          bool, std::is_same<paddle::platform::complex128,
                             typename std::remove_cv<
                                 paddle::platform::complex128>::type>::value> {
};
template <>
struct is_signed<paddle::platform::complex128> {
  static const bool value = false;
};

template <>
struct is_unsigned<paddle::platform::complex128> {
  static const bool value = false;
};

inline bool isnan(const paddle::platform::complex128& a) {
  return paddle::platform::isnan(a);
}

inline bool isinf(const paddle::platform::complex128& a) {
  return paddle::platform::isinf(a);
}

}  // namespace std
namespace Eigen {

using complex128 = paddle::platform::complex128;

template <>
struct NumTraits<complex128> : GenericNumTraits<std::complex<double>> {
  typedef double Real;
  typedef typename NumTraits<double>::Literal Literal;
  enum {
    IsComplex = 1,
    RequireInitialization = NumTraits<double>::RequireInitialization,
    ReadCost = 2 * NumTraits<double>::ReadCost,
    AddCost = 2 * NumTraits<Real>::AddCost,
    MulCost = 4 * NumTraits<Real>::MulCost + 2 * NumTraits<Real>::AddCost
  };

  EIGEN_DEVICE_FUNC
  static inline Real epsilon() { return NumTraits<Real>::epsilon(); }
  EIGEN_DEVICE_FUNC
  static inline Real dummy_precision() {
    return NumTraits<Real>::dummy_precision();
  }
  EIGEN_DEVICE_FUNC
  static inline int digits10() { return NumTraits<Real>::digits10(); }
};
namespace numext {

template <>
HOSTDEVICE inline bool(isnan)(const complex128& a) {
  return (std::isnan)(a.real) || (std::isnan)(a.imag);
}

template <>
HOSTDEVICE inline bool(isinf)(const complex128& a) {
  return (std::isinf)(a.real) || (std::isinf)(a.imag);
}

template <>
HOSTDEVICE inline bool(isfinite)(const complex128& a) {
  return (std::isfinite)(a.real) || (std::isfinite)(a.imag);
}

template <>
HOSTDEVICE inline complex128 exp(const complex128& a) {
  double com = ::expf(a.real);
  double res_real = com * ::cosf(a.imag);
  double res_imag = com * ::sinf(a.imag);
  return complex128(res_real, res_imag);
}

template <>
HOSTDEVICE inline complex128 log(const complex128& a) {
  std::complex<double> a_(a.real, a.imag);
  return complex128(std::log(a_));
}

template <>
HOSTDEVICE inline complex128 tanh(const complex128& a) {
  std::complex<double> a_(a.real, a.imag);
  return complex128(std::tanh(a_));
}

template <>
HOSTDEVICE inline complex128 sqrt(const complex128& a) {
  std::complex<double> a_(a.real, a.imag);
  return complex128(std::sqrt(a_));
}

template <>
HOSTDEVICE inline complex128 ceil(const complex128& a) {
  return complex128(::ceilf(a.real), ::ceilf(a.imag));
}

template <>
HOSTDEVICE inline complex128 floor(const complex128& a) {
  return complex128(::floorf(a.real), ::floor(a.imag));
}

template <>
HOSTDEVICE inline complex128 round(const complex128& a) {
  return complex128(::roundf(a.real), ::roundf(a.imag));
}

template <>
HOSTDEVICE inline complex128 pow(const complex128& a, const complex128& b) {
  std::complex<double> a_(a.real, a.imag);
  std::complex<double> b_(b.real, b.imag);
  return complex128(std::pow(a_, b_));
}

template <>
HOSTDEVICE inline double abs(const complex128& a) {
  return ::hypotf(a.real, a.imag);
}

}  // namespace numext
}  // namespace Eigen

#define MKL_Complex16 paddle::platform::complex128
