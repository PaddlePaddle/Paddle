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
/*! \file
    \brief Define basic numeric operators with specializations for Array<T, N>. SIMD-ize where possible.

    This is inspired by the Standard Library's <functional> header.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/complex.h"
#include "cutlass/quaternion.h"
#include "cutlass/array.h"
#include "cutlass/half.h"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct absolute_value_op {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return abs(lhs);
  }
};

template <typename T>
struct plus {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs += rhs;
    return lhs;
  }
};

template <typename T>
struct minus {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs -= rhs;
    return lhs;
  }
};

template <typename T>
struct multiplies {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs *= rhs;
    return lhs;
  }
};

template <typename T>
struct multiplies<Quaternion<T>> {
  CUTLASS_HOST_DEVICE
  Quaternion<T> operator()(Quaternion<T> lhs, Quaternion<T> const &rhs) const {
    lhs = lhs * rhs;
    return lhs;
  }
};

/// Squares with optional conversion
template <typename T, typename Output = T>
struct square {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Returns the magnitude squared of an element.
template <typename T, typename Output = T>
struct magnitude_squared {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Squares with optional conversion
template <typename T, typename Output>
struct magnitude_squared<complex<T>, Output> {
  CUTLASS_HOST_DEVICE
  Output operator()(complex<T> lhs) const {
    multiplies<Output> mul_op;

    Output y_r = Output(lhs.real());
    Output y_i = Output(lhs.imag());

    return mul_op(y_r, y_r) + mul_op(y_i, y_i);
  }
};

/// Squares with optional conversion
template <typename T, typename Output>
struct magnitude_squared<Quaternion<T>, Output> {
  CUTLASS_HOST_DEVICE
  Output operator()(Quaternion<T> lhs) const {
    multiplies<Output> mul_op;

    Output y_w = Output(lhs.w());
    Output y_x = Output(lhs.x());
    Output y_y = Output(lhs.y());
    Output y_z = Output(lhs.z());

    return mul_op(y_w, y_w) + mul_op(y_x, y_x) + mul_op(y_y, y_y) + \
           mul_op(y_z, y_z);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T>
struct square_difference {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T>
struct magnitude_squared_difference {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output>
struct magnitude_squared_difference<complex<T>, Output> {
  CUTLASS_HOST_DEVICE
  Output operator()(complex<T> lhs, complex<T> rhs) const {
    multiplies<Output> mul_op;

    Output y_r = Output(lhs.real()) - Output(rhs.real());
    Output y_i = Output(lhs.imag()) - Output(rhs.imag());

    return mul_op(y_r, y_r) + mul_op(y_i, y_i);
  }
};

template <typename T>
struct divides {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs /= rhs;
    return lhs;
  }
};


template <typename T>
struct negate {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return -lhs;
  }
};

/// Greater equal 
template <typename T>
struct greater_equal {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs >= rhs);
  }
};

/// Greater  
template <typename T>
struct greater {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs > rhs);
  }
};

/// Less equal 
template <typename T>
struct less_equal {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs <= rhs);
  }
};

/// Less  
template <typename T>
struct less {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs < rhs);
  }
};

template <typename T>
struct maximum {

  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    return (lhs < rhs ? rhs : lhs);
  }
};

template <>
struct maximum<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const {
    return fmaxf(lhs, rhs);
  }
};

template <typename T>
struct minimum {

  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    return (rhs < lhs ? rhs : lhs);
  }
};

template <>
struct minimum<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const {
    return fminf(lhs, rhs);
  }
};

/// Fused multiply-add
template <typename A, typename B = A, typename C = A>
struct multiply_add {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    return C(a) * C(b) + c;
  }
};

/// Fused multiply-add
template <typename A, typename B = A, typename C = A>
struct multiply_add_relu0 {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    maximum<C> mx;
    return mx(C(a) * C(b) + c, C(0));
  }
};

/// Fused multiply-add
template <typename T>
struct and_add {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a & b) + c);
  }
};


/// Fused multiply-add
template <typename T>
struct xor_add {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a ^ b) + c);
  }
};

template <typename T>
struct conjugate {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return a;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct logical_and {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return ((a && b) ? T(1) : T());
  }
};

template <typename T>
struct logical_or {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return ((a || b) ? T(1) : T());
  }
};

template <typename T>
struct logical_not {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return T(!(a));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct bit_and {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a & b;
  }
};

template <typename T>
struct bit_or {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a | b;
  }
};

template <typename T>
struct bit_not {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return ~a;
  }
};

template <typename T>
struct bit_xor {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a ^ b;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Partial specializations for Arrays
template <int N>
struct bit_and<Array<uint1b_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<uint1b_t, N> operator()(Array<uint1b_t, N> const &a, Array<uint1b_t, N> const &b) const {
    using ArrayType = Array<uint1b_t, N>;
    using Storage = typename ArrayType::Storage;
    ArrayType result;

    Storage *result_data = result.raw_data();
    Storage const *a_data = a.raw_data();
    Storage const *b_data = b.raw_data();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ArrayType::kStorageElements; ++i) {
      result_data[i] = (a_data[i] & b_data[i]);
    }

    return result;
  }
};

// Partial specializations for Arrays
template <int N>
struct bit_or<Array<uint1b_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<uint1b_t, N> operator()(Array<uint1b_t, N> const &a, Array<uint1b_t, N> const &b) const {
    using ArrayType = Array<uint1b_t, N>;
    using Storage = typename ArrayType::Storage;
    ArrayType result;

    Storage *result_data = result.raw_data();
    Storage const *a_data = a.raw_data();
    Storage const *b_data = b.raw_data();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ArrayType::kStorageElements; ++i) {
      result_data[i] = (a_data[i] | b_data[i]);
    }

    return result;
  }
};

// Partial specializations for Arrays
template <int N>
struct bit_not<Array<uint1b_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<uint1b_t, N> operator()(Array<uint1b_t, N> const &a) const {
    using ArrayType = Array<uint1b_t, N>;
    using Storage = typename ArrayType::Storage;
    ArrayType result;

    Storage *result_data = result.raw_data();
    Storage const *a_data = a.raw_data();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ArrayType::kStorageElements; ++i) {
      result_data[i] = (~a_data[i]);
    }

    return result;
  }
};

// Partial specializations for Arrays
template <int N>
struct bit_xor<Array<uint1b_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<uint1b_t, N> operator()(Array<uint1b_t, N> const &a, Array<uint1b_t, N> const &b) const {
    using ArrayType = Array<uint1b_t, N>;
    using Storage = typename ArrayType::Storage;
    ArrayType result;

    Storage *result_data = result.raw_data();
    Storage const *a_data = a.raw_data();
    Storage const *b_data = b.raw_data();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ArrayType::kStorageElements; ++i) {
      result_data[i] = (a_data[i] ^ b_data[i]);
    }

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct conjugate<complex<T>>  {
  CUTLASS_HOST_DEVICE
  complex<T> operator()(complex<T> const &a) const {
    return conj(a);
  }
};

template <typename T, int N>
struct conjugate<Array<T, N> >  {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a) const {

    conjugate<T> conj_op;

    Array<T, N> ca;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      ca[i] = conj_op(a[i]);
    }
    return ca;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specialization for complex<T> to target four scalar fused multiply-adds.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Fused multiply-add
template <typename T>
struct multiply_add<complex<T>, complex<T>, complex<T>> {
  CUTLASS_HOST_DEVICE
  complex<T> operator()(
    complex<T> const &a, 
    complex<T> const &b, 
    complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a.real() * b.real();
    real += -a.imag() * b.imag();
    imag += a.real() * b.imag();
    imag += a.imag () * b.real();

    return complex<T>{
      real,
      imag
    };
  }
};

/// Fused multiply-add
template <typename T>
struct multiply_add<complex<T>, T, complex<T>> {
  CUTLASS_HOST_DEVICE
  complex<T> operator()(
    complex<T> const &a, 
    T const &b, 
    complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a.real() * b;
    imag += a.imag () * b;

    return complex<T>{
      real,
      imag
    };
  }
};

/// Fused multiply-add
template <typename T>
struct multiply_add<T, complex<T>, complex<T>> {
  CUTLASS_HOST_DEVICE
  complex<T> operator()(
    T const &a, 
    complex<T> const &b, 
    complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a * b.real();
    imag += a * b.imag();

    return complex<T>{
      real,
      imag
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<T, N>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
struct absolute_value_op< Array<T, N> > {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs) const {

    Array<T, N> result;
    absolute_value_op<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct plus<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    
    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};
template <typename T, int N>
struct minus<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    minus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    
    Array<T, N> result;
    minus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    minus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct multiplies<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    
    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct divides<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    divides<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    
    Array<T, N> result;
    divides<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    divides<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct maximum<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    maximum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    
    Array<T, N> result;
    maximum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    maximum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct minimum<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  static T scalar_op(T const &lhs, T const &rhs) {
    return (rhs < lhs ? rhs : lhs);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    minimum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    
    Array<T, N> result;
    minimum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {
    
    Array<T, N> result;
    minimum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct negate<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs) const {
    
    Array<T, N> result;
    negate<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i]);
    }

    return result;
  }
};

/// Fused multiply-add
template <typename T, int N>
struct multiply_add<Array<T, N>, Array<T, N>, Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) const {
    
    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], b[i], c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, T const &scalar, Array<T, N> const &c) const {
    
    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], scalar, c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &b, Array<T, N> const &c) const {
    
    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, b[i], c[i]);
    }

    return result;
  }
};

/// Fused multiply-add-relu0
template <typename T, int N>
struct multiply_add_relu0<Array<T, N>, Array<T, N>, Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) const {
    
    Array<T, N> result;
    multiply_add<T> scalar_op;
    maximum<T> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(scalar_op(a[i], b[i], c[i]), T(0));
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, T const &scalar, Array<T, N> const &c) const {
    
    Array<T, N> result;
    multiply_add<T> scalar_op;
    maximum<T> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(scalar_op(a[i], scalar, c[i]), T(0));
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &b, Array<T, N> const &c) const {
    
    Array<T, N> result;
    multiply_add<T> scalar_op;
    maximum<T> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(scalar_op(scalar, b[i], c[i]), T(0));
    }

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<half_t, N> targeting SIMD instructions in device code.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
struct plus<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hadd2(lhs_ptr[i], rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hadd(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] + rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hadd2(lhs_pair, rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hadd(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs + rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hadd2(lhs_ptr[i], rhs_pair);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half d_residual = __hadd(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] + rhs;
    }
    #endif

    return result;
  }
};

template <int N>
struct minus<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hsub2(lhs_ptr[i], rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hsub(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] - rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hsub2(lhs_pair, rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hsub(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs - rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hsub2(lhs_ptr[i], rhs_pair);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half d_residual = __hsub(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] - rhs;
    }
    #endif

    return result;
  }
};

template <int N>
struct multiplies<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmul2(lhs_ptr[i], rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hmul(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] * rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmul2(lhs_pair, rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = __hmul(
        reinterpret_cast<__half const &>(lhs), 
        b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs * rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmul2(lhs_ptr[i], rhs_pair);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);

      __half d_residual = __hmul(
        a_residual_ptr[N - 1], 
        reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] * rhs;
    }
    #endif

    return result;
  }
};

template <int N>
struct divides<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __h2div(lhs_ptr[i], rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = __hdiv(
        a_residual_ptr[N - 1], 
        b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] / rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __h2div(lhs_pair, rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = __hdiv(
        reinterpret_cast<__half const &>(lhs), 
        b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs / rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __h2div(lhs_ptr[i], rhs_pair);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);

      __half d_residual = __hdiv(
        a_residual_ptr[N - 1], 
        reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] / rhs;
    }
    #endif

    return result;
  }
};

template <int N>
struct negate<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *source_ptr = reinterpret_cast<__half2 const *>(&lhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hneg2(source_ptr[i]);
    }

    if (N % 2) {
      half_t x = lhs[N - 1];
      __half lhs_val = -reinterpret_cast<__half const &>(x);
      result[N - 1] = reinterpret_cast<half_t const &>(lhs_val);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = -lhs[i];
    }
    #endif

    return result;
  }
};

/// Fused multiply-add
template <int N>
struct multiply_add<Array<half_t, N>, Array<half_t, N>, Array<half_t, N>> {

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a, 
    Array<half_t, N> const &b, 
    Array<half_t, N> const &c) const {
    
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_ptr[i], b_ptr[i], c_ptr[i]);
    }

    if (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);

      __half d_residual = __hfma(
        a_residual_ptr[N - 1], 
        b_residual_ptr[N - 1], 
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    half_t const &a, 
    Array<half_t, N> const &b, 
    Array<half_t, N> const &c) const {
    
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 a_pair = __half2half2(reinterpret_cast<__half const &>(a));
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_pair, b_ptr[i], c_ptr[i]);
    }

    if (N % 2) {

      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);
      __half d_residual = __hfma(
        reinterpret_cast<__half const &>(a), 
        b_residual_ptr[N - 1], 
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a, b[i], c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a, 
    half_t const &b, 
    Array<half_t, N> const &c) const {
    
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 b_pair = __half2half2(reinterpret_cast<__half const &>(b));
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_ptr[i], b_pair, c_ptr[i]);
    }

    if (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);

      __half d_residual = __hfma(
        a_residual_ptr[N - 1], 
        reinterpret_cast<__half const &>(b), 
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b, c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a, 
    Array<half_t, N> const &b, 
    half_t const &c) const {
    
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 c_pair = __half2half2(reinterpret_cast<__half const &>(c));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_ptr[i], b_ptr[i], c_pair);
    }

    if (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);

      __half d_residual = __hfma(
        a_residual_ptr[N - 1], 
        b_residual_ptr[N - 1], 
        reinterpret_cast<__half const &>(c));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c);
    }
    #endif

    return result;
  }
};

/// Fused multiply-add-relu0
template <int N>
struct multiply_add_relu0<Array<half_t, N>, Array<half_t, N>, Array<half_t, N>> {

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a, 
    Array<half_t, N> const &b, 
    Array<half_t, N> const &c) const {
    
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2_relu(a_ptr[i], b_ptr[i], c_ptr[i]);
    }

    if (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);

      __half d_residual = __hfma_relu(
        a_residual_ptr[N - 1], 
        b_residual_ptr[N - 1], 
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;
    maximum<half_t> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(op(a[i], b[i], c[i]), (half_t)0);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    half_t const &a, 
    Array<half_t, N> const &b, 
    Array<half_t, N> const &c) const {
    
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 a_pair = __half2half2(reinterpret_cast<__half const &>(a));
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2_relu(a_pair, b_ptr[i], c_ptr[i]);
    }

    if (N % 2) {

      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);
      __half d_residual = __hfma_relu(
        reinterpret_cast<__half const &>(a), 
        b_residual_ptr[N - 1], 
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;
    maximum<half_t> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(op(a, b[i], c[i]), half_t(0));
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a, 
    half_t const &b, 
    Array<half_t, N> const &c) const {
    
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 b_pair = __half2half2(reinterpret_cast<__half const &>(b));
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2_relu(a_ptr[i], b_pair, c_ptr[i]);
    }

    if (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);

      __half d_residual = __hfma_relu(
        a_residual_ptr[N - 1], 
        reinterpret_cast<__half const &>(b), 
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;
    maximum<half_t> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(op(a[i], b, c[i]), half_t(0));
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a, 
    Array<half_t, N> const &b, 
    half_t const &c) const {
    
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 c_pair = __half2half2(reinterpret_cast<__half const &>(c));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2_relu(a_ptr[i], b_ptr[i], c_pair);
    }

    if (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);

      __half d_residual = __hfma_relu(
        a_residual_ptr[N - 1], 
        b_residual_ptr[N - 1], 
        reinterpret_cast<__half const &>(c));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;
    maximum<half_t> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(op(a[i], b[i], c), half_t(0));
    }
    #endif

    return result;
  }
};

template <int N>
struct minimum<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmin2(lhs_ptr[i], rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = __hmin(
        a_residual_ptr[N - 1], 
        b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = (rhs[i] < lhs[i] ? rhs[i] : lhs[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmin2(lhs_pair, rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = __hmin(
        reinterpret_cast<__half const &>(lhs), 
        b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = (rhs[i] < lhs ? rhs[i] : lhs);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmin2(lhs_ptr[i], rhs_pair);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);

      __half d_residual = __hmin(
        a_residual_ptr[N - 1], 
        reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = (rhs < lhs[i] ? rhs : lhs[i]);
    }
    #endif

    return result;
  }
};

template <int N>
struct maximum<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmax2(lhs_ptr[i], rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = __hmax(
        a_residual_ptr[N - 1], 
        b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = (lhs[i] < rhs[i] ? rhs[i] : lhs[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmax2(lhs_pair, rhs_ptr[i]);
    }

    if (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = __hmax(
        reinterpret_cast<__half const &>(lhs), 
        b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = (lhs < rhs[i] ? rhs[i] : lhs);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmax2(lhs_ptr[i], rhs_pair);
    }

    if (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);

      __half d_residual = __hmax(
        a_residual_ptr[N - 1], 
        reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = (lhs[i] < rhs ? rhs : lhs[i]);
    }
    #endif

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Fused multiply-add
template <int N>
struct multiply_add<Array<bfloat16_t, N>, Array<bfloat16_t, N>, Array<bfloat16_t, N>> {

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(
    Array<bfloat16_t, N> const &a, 
    Array<bfloat16_t, N> const &b, 
    Array<bfloat16_t, N> const &c) const {
    
    Array<bfloat16_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);
    unsigned const *a_ptr = reinterpret_cast<unsigned const *>(&a);
    unsigned const *b_ptr = reinterpret_cast<unsigned const *>(&b);
    unsigned const *c_ptr = reinterpret_cast<unsigned const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm ("fma.rn.bf16x2 %0, %1, %2, %3;\n" 
        : "=r"(result_ptr[i]) 
        : "r"(a_ptr[i]), "r"(b_ptr[i]), "r"(c_ptr[i])
      );
    }

    if (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm ("fma.rn.bf16 %0, %1, %2, %3;\n" 
        : "=h"(result_ptr[N - 1]) 
        : "h"(a_residual_ptr[N - 1]), "h"(b_residual_ptr[N - 1]), "h"(c_residual_ptr[N - 1])
      );
    }

    #else

    multiply_add<bfloat16_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(
    bfloat16_t const &a, 
    Array<bfloat16_t, N> const &b, 
    Array<bfloat16_t, N> const &c) const {
    
    Array<bfloat16_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);

    unsigned const *b_ptr = reinterpret_cast<unsigned const *>(&b);
    unsigned const *c_ptr = reinterpret_cast<unsigned const *>(&c);

    unsigned a_packed = static_cast<unsigned>(a.raw());
    a_packed = (a_packed | (a_packed << 16));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm ("fma.rn.bf16x2 %0, %1, %2, %3;\n" 
        : "=r"(result_ptr[i]) 
        : "r"(a_packed), "r"(b_ptr[i]), "r"(c_ptr[i])
      );
    }

    if (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm ("fma.rn.bf16 %0, %1, %2, %3;\n" 
        : "=h"(result_ptr[N - 1]) 
        : "h"(a_residual_ptr[0]), "h"(b_residual_ptr[N - 1]), "h"(c_residual_ptr[N - 1])
      );
    }

    #else

    multiply_add<bfloat16_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a, b[i], c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(
    Array<bfloat16_t, N> const &a, 
    bfloat16_t const &b, 
    Array<bfloat16_t, N> const &c) const {
    
    Array<bfloat16_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);
    
    unsigned const *a_ptr = reinterpret_cast<unsigned const *>(&a);
    unsigned const *c_ptr = reinterpret_cast<unsigned const *>(&c);

    unsigned b_packed = static_cast<unsigned>(b.raw());
    b_packed = (b_packed | (b_packed << 16));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm ("fma.rn.bf16x2 %0, %1, %2, %3;\n" 
        : "=r"(result_ptr[i]) 
        : "r"(a_ptr[i]), "r"(b_packed), "r"(c_ptr[i])
      );
    }

    if (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm ("fma.rn.bf16 %0, %1, %2, %3;\n" 
        : "=h"(result_ptr[N - 1]) 
        : "h"(a_residual_ptr[N - 1]), "h"(b_residual_ptr[0]), "h"(c_residual_ptr[N - 1])
      );
    }

    #else

    multiply_add<bfloat16_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b, c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(
    Array<bfloat16_t, N> const &a, 
    Array<bfloat16_t, N> const &b, 
    bfloat16_t const &c) const {
    
    Array<bfloat16_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);
    
    unsigned const *a_ptr = reinterpret_cast<unsigned const *>(&a);
    unsigned const *b_ptr = reinterpret_cast<unsigned const *>(&b);

    unsigned c_packed = static_cast<unsigned>(c.raw());
    c_packed = (c_packed | (c_packed << 16));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm ("fma.rn.bf16x2 %0, %1, %2, %3;\n" 
        : "=r"(result_ptr[i]) 
        : "r"(a_ptr[i]), "r"(b_ptr[i]), "r"(c_packed)
      );
    }

    if (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm ("fma.rn.bf16 %0, %1, %2, %3;\n" 
        : "=h"(result_ptr[N - 1]) 
        : "h"(a_residual_ptr[N - 1]), "h"(b_residual_ptr[N - 1]), "h"(c_residual_ptr[0])
      );
    }

    #else

    multiply_add<bfloat16_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c);
    }
    #endif

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////


template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator+(Array<T, N> const &lhs, Array<T, N> const &rhs) {
  plus<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator-(Array<T, N> const &lhs, Array<T, N> const &rhs) {
  minus<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator-(Array<T, N> const &lhs) {
  negate<Array<T, N>> op;
  return op(lhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator*(Array<T, N> const &lhs, Array<T, N> const &rhs) {
  multiplies<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator*(T lhs, Array<T, N> const &rhs) {
  multiplies<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator*(Array<T, N> const &lhs, T rhs) {
  multiplies<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator/(Array<T, N> const &lhs, Array<T, N> const &rhs) {
  divides<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> fma(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) {
  multiply_add<Array<T, N>> op;
  return op(a, b, c);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> fma(T a, Array<T, N> const &b, Array<T, N> const &c) {
  multiply_add<Array<T, N>> op;
  return op(a, b, c);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> fma(Array<T, N> const &a, T b, Array<T, N> const &c) {
  multiply_add<Array<T, N>> op;
  return op(a, b, c);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> fma(Array<T, N> const &a, Array<T, N> const &b, T c) {
  multiply_add<Array<T, N>> op;
  return op(a, b, c);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Quaternion<T> fused multiply-add
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct multiply_add<Quaternion<T>, Quaternion<T>, Quaternion<T>> {
  CUTLASS_HOST_DEVICE
  Quaternion<T> operator()(
    Quaternion<T> const &a,
    Quaternion<T> const &b,
    Quaternion<T> const &c) const {

    T x = c.x();
    T y = c.y();
    T z = c.z();
    T w = c.w();

    x += a.w() * b.x();
    x += b.w() * a.x();
    x += a.y() * b.z();
    x += -a.z() * b.y(),

    y += a.w() * b.y();
    y += b.w() * a.y();
    y += a.z() * b.x();
    y += -a.x() * b.z();

    z += a.w() * b.z();
    z += b.w() * a.z();
    z += a.x() * b.y();
    z += -a.y() * b.x();

    w += a.w() * b.w();
    w += -a.x() * b.x();
    w += -a.y() * b.y();
    w += -a.z() * b.z();
    
    return cutlass::make_Quaternion(x, y, z, w);

  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
