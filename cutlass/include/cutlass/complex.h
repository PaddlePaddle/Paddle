/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cuComplex.h>

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
#else
#include <cstdint>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/real.h"

#include "cutlass/bfloat16.h"
#include "cutlass/tfloat32.h"

#include "cutlass/fast_math.h"

#if !defined(__CUDACC_RTC__)
#include <iosfwd>
#endif

namespace cutlass {

//////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumeraed type describing a transformation on a complex value.
enum class ComplexTransform {
  kNone,
  kConjugate
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines ComplexTransform inversions
template <ComplexTransform kTransform>
struct InvertComplexTransform;

/// Invert ComplexTransform from kNone to kConjugate
template <>
struct InvertComplexTransform<ComplexTransform::kNone> {
  static ComplexTransform const transform = ComplexTransform::kConjugate;
};

/// Invert ComplexTransform from kConjugate to kNone
template <>
struct InvertComplexTransform<ComplexTransform::kConjugate> {
  static ComplexTransform const transform = ComplexTransform::kNone;
};
/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//
// Accessors for CUDA complex types
//

#if !defined(__CUDACC_RTC__)
/// Returns the real part of the complex number
CUTLASS_HOST_DEVICE
float const &real(cuFloatComplex const &z) { return z.x; }

/// Returns the real part of the complex number
CUTLASS_HOST_DEVICE
float &real(cuFloatComplex &z) { return z.x; }

/// Returns the real part of the complex number
CUTLASS_HOST_DEVICE
double const &real(cuDoubleComplex const &z) { return z.x; }

/// Returns the real part of the complex number
CUTLASS_HOST_DEVICE
double &real(cuDoubleComplex &z) { return z.x; }

/// Returns the imaginary part of the complex number
CUTLASS_HOST_DEVICE
float const &imag(cuFloatComplex const &z) { return z.y; }

/// Returns the imaginary part of the complex number
CUTLASS_HOST_DEVICE
float &imag(cuFloatComplex &z) { return z.y; }

/// Returns the imaginary part of the complex number
CUTLASS_HOST_DEVICE
double const &imag(cuDoubleComplex const &z) { return z.y; }

/// Returns the imaginary part of the complex number
CUTLASS_HOST_DEVICE
double &imag(cuDoubleComplex &z) { return z.y; }
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Class for representing and manipulating complex numbers with conversions from built-in CUDA
/// complex types.

template <typename T>
class complex
{
 public:
  /// Type alias for scalar type

 private:
  //
  // Data members
  //

  /// Real part
  T _real;

  /// Imaginary part
  T _imag;

 public:

//
// Methods
//

/// Constructor
  CUTLASS_HOST_DEVICE
  complex(T r = T(0)) : _real(r), _imag(T(0)) {}

/// Constructor
  CUTLASS_HOST_DEVICE
  complex(T r, T i) : _real(r), _imag(i) {}
  //
/// Constructor
  template<typename A>
  CUTLASS_HOST_DEVICE
  complex(complex<A> const &z) : _real(static_cast<T>(z.real())), _imag(static_cast<T>(z.imag())) {}


  #if !defined(__CUDACC_RTC__)
  /// Conversion from cuFloatComplex
  CUTLASS_HOST_DEVICE
  complex(cuFloatComplex const &z) : _real(static_cast<T>(cuCrealf(z))), _imag(static_cast<T>(cuCimagf(z))) {}

  /// Conversion from cuDoubleComplex
  CUTLASS_HOST_DEVICE
  complex(cuDoubleComplex const &z) : _real(static_cast<T>(cuCreal(z))), _imag(static_cast<T>(cuCimag(z))) {}
  #endif

  /// Assignment
  template<typename A>
  CUTLASS_HOST_DEVICE
  complex<T>& operator=(complex<A> const &z)
  {
    _real = static_cast<T>(z.real());
    _imag = static_cast<T>(z.imag());
    return *this;
  }

  /// Equality operator
  CUTLASS_HOST_DEVICE bool operator==(complex<T> const &rhs) const {
    return this->real() == rhs.real() && this->imag() == rhs.imag();
  }

  /// Inequality operator
  CUTLASS_HOST_DEVICE bool operator!=(complex<T> const &rhs) const {
    return !(*this == rhs);
  }

  /// Addition
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator+(complex<A> const &rhs) const {
    return complex<T>(this->real() + rhs.real(), this->imag() + rhs.imag());
  }

  /// Subtraction
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator-(complex<A> const &rhs) const {
    return complex<T>(this->real() - rhs.real(), this->imag() - rhs.imag());
  }

  /// Multiplication
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator*(complex<A> const &rhs) const {
    return complex<T>(this->real() * rhs.real() - this->imag() * rhs.imag(),
                      this->real() * rhs.imag() + this->imag() * rhs.real());
  }

  /// Scalar Multiplication
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator*(A const &s) const {
    return complex<T>(this->real() * s, this->imag() * s);
  }

  /// Division
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator/(complex<A> const &rhs) const {
    T d = T(rhs.real() * rhs.real() + rhs.imag() * rhs.imag());

    return complex<T>(
      (real() * rhs.real() + imag() * rhs.imag()) / d,
      (imag() * rhs.real() - real() * rhs.imag()) / d
    );
  }

  /// Scalar Division
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator/(A const &s) const {
    return complex<T>(this->real() / s, this->imag() / s);
  }

  /// Addition
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> &operator+=(complex<A> const &rhs) {
      *this = *this + rhs;
      return *this;
  }

  /// Subtraction
  template <typename A>
  CUTLASS_HOST_DEVICE complex<T> &operator-=(complex<A> const &rhs) {
      *this = *this - rhs;
      return *this;
  }

  /// Multiplication
  template <typename A>
  CUTLASS_HOST_DEVICE complex<T> &operator*=(complex<A> const &rhs) {
      *this = *this * rhs;
      return *this;
  }

  /// Scalar multiplication
  template <typename A>
  CUTLASS_HOST_DEVICE complex<T> &operator*=(A s) {
      *this = *this * s;
      return *this;
  }

  /// Division
  template <typename A>
  CUTLASS_HOST_DEVICE complex<T> &operator/=(complex<A> const &rhs) {
      *this = *this / rhs;
      return *this;
  }

  /// Accesses the real part of the complex number
  CUTLASS_HOST_DEVICE
  T const &real() const { return _real; }

  /// Accesses the real part of the complex number
  CUTLASS_HOST_DEVICE
  T &real() { return _real; }

  /// Accesses the imaginary part of the complex number
  CUTLASS_HOST_DEVICE
  T const &imag() const { return _imag; }

  /// Accesses the imaginary part of the complex number
  CUTLASS_HOST_DEVICE
  T &imag() { return _imag; }


  #if !defined(__CUDACC_RTC__)
  /// Converts to cuFloatComplex
  CUTLASS_HOST_DEVICE
  explicit operator cuFloatComplex() const { return make_cuFloatComplex(float(real()), float(imag())); }

  /// Converts to cuDoubleComplex
  CUTLASS_HOST_DEVICE
  explicit operator cuDoubleComplex() const { return make_cuDoubleComplex(real(), imag()); }
  #endif
};

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// Accessors for complex template
//

/// Returns the real part of the complex number
template <typename T>
CUTLASS_HOST_DEVICE T const &real(complex<T> const &z) {
  return z.real();
}

/// Returns the real part of the complex number
template <typename T>
CUTLASS_HOST_DEVICE T &real(complex<T> &z) {
  return z.real();
}

/// Returns the imaginary part of the complex number
template <typename T>
CUTLASS_HOST_DEVICE T const &imag(complex<T> const &z) {
  return z.imag();
}

/// Returns the imaginary part of the complex number
template <typename T>
CUTLASS_HOST_DEVICE T &imag(complex<T> &z) {
  return z.imag();
}

/// Returns the real part of the real number
template <typename T>
CUTLASS_HOST_DEVICE T const &real(T const &r) {
  return r;
}

/// Returns the real part of the real number
template <typename T>
CUTLASS_HOST_DEVICE T &real(T &r) {
  return r;
}

/// Returns the imaginary part of the real number
template <typename T>
CUTLASS_HOST_DEVICE T const &imag(T const &r) {
  return T();
}

/// Returns the imaginary part of the complex number
template <typename T>
CUTLASS_HOST_DEVICE T &imag(T &r) {
  return T();
}

//
// Output operators
//

#if !defined(__CUDACC_RTC__)
template <typename T>
std::ostream &operator<<(std::ostream &out, complex<T> const &z) {
  T _r = real(z);
  T _i = imag(z);

  if (bool(_i)) {
    return out << _r << "+i" << _i;
  }
  return out << _r;
}
#endif

//
// Non-member operators defined for complex types
//


//
// Non-member functions defined for complex numbers
//

/// Returns the magnitude of the complex number
template <typename T>
CUTLASS_HOST_DEVICE T abs(complex<T> const &z) {
  return sqrt(norm(z));
}

/// Returns the magnitude of the complex number
template <typename T>
CUTLASS_HOST_DEVICE T arg(complex<T> const &z) {
  return atan2(imag(z), real(z));
}

/// Returns the squared magnitude of a real number
template <typename T>
CUTLASS_HOST_DEVICE T norm(T const &z) {
    return z * z;
}

/// Returns the squared magnitude of a real number
template <>
CUTLASS_HOST_DEVICE int8_t norm(int8_t const &z) {
    return static_cast<int8_t>(z * z);
}

/// Returns the squared magnitude of a complex number
template <typename T>
CUTLASS_HOST_DEVICE double norm(complex<T> const &z) {
  return real(z) * real(z) + imag(z) * imag(z);
}

/// Norm-accumulate calculation
template <typename T, typename R>
CUTLASS_HOST_DEVICE R norm_accumulate(T const &x, R const & accumulator) {
  return accumulator + static_cast<R>(x) * static_cast<R>(x);
}

/// Norm accumulate specialized for complex types
template <typename T, typename R>
CUTLASS_HOST_DEVICE R norm_accumulate(complex<T> const &z, R const &accumulator) {
  return accumulator + static_cast<R>(real(z)) * static_cast<R>(real(z)) + 
    static_cast<R>(imag(z)) * static_cast<R>(imag(z));
}

/// Returns the complex conjugate
CUTLASS_HOST_DEVICE float conj(float const &z) {
  return z;
}

/// Returns the complex conjugate
CUTLASS_HOST_DEVICE double conj(double const &z) {
  return z;
}

/// Returns the complex conjugate
template <typename T>
CUTLASS_HOST_DEVICE complex<T> conj(complex<T> const &z) {
  return complex<T>(real(z), -imag(z));
}
/// Indentity transform for non-complex types
template <typename T>
CUTLASS_HOST_DEVICE T conj(T const &z) {
    static_assert( !platform::is_same<T, cuComplex>::value &&
                   !platform::is_same<T, cuDoubleComplex>::value &&
                   !platform::is_same<T, cutlass::complex<double>>::value &&
                   !platform::is_same<T, cutlass::complex<float>>::value, "May not be a complex data type");
  return z;
}

/// Projects the complex number z onto the Riemann sphere
template <typename T>
CUTLASS_HOST_DEVICE complex<T> proj(complex<T> const &z) {
  T d = real(z) * real(z) + imag(z) * imag(z) + T(1);
  return complex<T>((T(2) * real(z)) / d, (T(2) * imag(z)) / d);
}

/// Returns a complex number with magnitude r and phase theta
template <typename T>
CUTLASS_HOST_DEVICE complex<T> polar(T const &r, T const &theta = T()) {
  return complex<T>(r * cos(theta), r * sin(theta));
}

/// Computes the complex exponential of z.
template <typename T>
CUTLASS_HOST_DEVICE complex<T> exp(complex<T> const &z) {
  return complex<T>(fast_exp(real(z)) * fast_cos(imag(z)), fast_exp(real(z)) * fast_sin(imag(z)));
}

/// Computes the log of z
template <typename T>
CUTLASS_HOST_DEVICE complex<T> log(complex<T> const &z) {
  return complex<T>(log(abs(z)), arg(z));
}

/// Computes the log base 10 of z
template <typename T>
CUTLASS_HOST_DEVICE complex<T> log10(complex<T> const &z) {
  return log(z) / T(log(T(10)));
}

/// Computes the square root of complex number z
template <typename T>
CUTLASS_HOST_DEVICE complex<T> sqrt(complex<T> const &z) {
  return sqrt(T(2)) / T(2) *
         complex<T>(sqrt(sqrt(norm(z)) + real(z)),
                    (imag(z) < 0 ? T(-1) : T(1)) * sqrt(sqrt(norm(z)) - real(z)));
}

/// Computes the cosine of complex z.
template <typename T>
CUTLASS_HOST_DEVICE complex<T> cos(complex<T> const &z) {
  return (exp(z) + exp(-z)) / T(2);
}

/// Computes the sin of complex z.
template <typename T>
CUTLASS_HOST_DEVICE complex<T> sin(complex<T> const &z) {
  return (exp(-z) - exp(z)) * complex<T>(T(0), T(1) / T(2));
}

/// Comparison 
template <typename T>
CUTLASS_HOST_DEVICE bool operator<(complex<T> const &lhs, complex<T> const &rhs) {
  //TODO
  return true; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex-valued type.
template <typename T>
struct RealType< complex<T> > {
  using Type = T;

  /// Number of elements
  static int const kExtent = 2;

CUTLASS_HOST_DEVICE
  static complex<T> from_real(double x) {
    return complex<T>(static_cast<T>(x));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_HOST_DEVICE
cutlass::complex<half_t> from_real<cutlass::complex<half_t> >(double r) {
  return cutlass::complex<half_t>(half_t(r));
}

template <>
CUTLASS_HOST_DEVICE
cutlass::complex<float> from_real<cutlass::complex<float> >(double r) {
  return cutlass::complex<float>(float(r));
}

template <>
CUTLASS_HOST_DEVICE
cutlass::complex<double> from_real<cutlass::complex<double> >(double r) {
  return cutlass::complex<double>(r);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct is_complex {
  static bool const value = false;
};

template <typename T>
struct is_complex<complex<T>> {
  static bool const value = true;
};

//////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

//////////////////////////////////////////////////////////////////////////////////////////////////
