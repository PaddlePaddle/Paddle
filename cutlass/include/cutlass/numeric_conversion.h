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
/*! 
    \file
    \brief Boost-like numeric conversion operator for CUTLASS numeric types
*/
#pragma once

#if !defined(__CUDACC_RTC__)
#include <cfenv>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/thread/unaryOp.h"

#include "cutlass/array.h"
#include "cutlass/half.h"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Floating-point rounding style similare to Standard Library's formats but supporting
/// additional rounding options.
enum class FloatRoundStyle {
  round_indeterminate,          ///< rounding mode unknown
  round_toward_zero,            ///< round toward zero
  round_to_nearest,             ///< round to nearest even
  round_toward_infinity,        ///< round toward infinity
  round_toward_neg_infinity,    ///< round toward negative infinity
  round_half_ulp_truncate,      ///< add 0.5ulp to integer representation then round toward zero
  round_half_ulp_trunc_dntz     ///< like round_half_ulp_truncate, except denorms are rounded *toward* zero
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
struct NumericConverter {

  using result_type = T;
  using source_type = S;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
    static result_type convert(source_type const & s) {

    return static_cast<result_type>(s);
  }

  CUTLASS_HOST_DEVICE
    result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float => int32_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__)
template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    return __float2int_rn(s);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  CUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    return __float2int_rz(s);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#elif !defined(__CUDACC_RTC__)

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TONEAREST);
    return (result_type)std::nearbyint(s);
  }

  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TOWARDZERO);
    return (result_type)std::nearbyint(s);
  }

  result_type operator()(source_type const &s) {
    return convert(s);
  }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float => int8_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__)
template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    int32_t intermediate = __float2int_rn(s);

    return static_cast<result_type>(intermediate);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  CUTLASS_DEVICE
  static result_type convert(source_type const & s) {

    int32_t intermediate = __float2int_rz(s);

    return static_cast<result_type>(intermediate);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#elif !defined(__CUDACC_RTC__)

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TONEAREST);
    int32_t intermediate =  (result_type)std::nearbyint(s);
    return static_cast<result_type>(intermediate);
  }

  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TOWARDZERO);
    int32_t intermediate =  (result_type)std::nearbyint(s);
    return static_cast<result_type>(intermediate);
  }

  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= half_t
template <typename T, FloatRoundStyle Round>
struct NumericConverter<T, T, Round> {

  using result_type = T;
  using source_type = T;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return s;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> half_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= half_t
template <FloatRoundStyle Round>
struct NumericConverter<float, half_t, Round> {

  using result_type = float;
  using source_type = half_t;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<float>(s);

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <>
struct NumericConverter<half_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = half_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<half_t>(s);

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <>
struct NumericConverter<half_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = half_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  /// Round toward zero
  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & flt) {

  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return half_t(__float2half_rz(flt));
  #else
    // software implementation rounds toward nearest even
    unsigned const& s = reinterpret_cast<unsigned const &>(flt);
    uint16_t sign = uint16_t((s >> 16) & 0x8000);
    int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
    int mantissa = s & 0x7fffff;
    uint16_t u = 0;

    if ((s & 0x7fffffff) == 0) {
      // sign-preserving zero
      return half_t::bitcast(sign);
    }

    if (exp > 15) {
      if (exp == 128 && mantissa) {
        // not a number
        u = 0x7fff;
      } else {
        // overflow to infinity
        u = sign | 0x7c00;
      }
      return half_t::bitcast(u);
    }

    if (exp >= -14) {
      // normal fp32 to normal fp16
      exp = uint16_t(exp + uint16_t(15));
      u = uint16_t(((exp & 0x1f) << 10));
      u = uint16_t(u | (mantissa >> 13));
    } else {
      // normal single-precision to subnormal half_t-precision representation
      int rshift = (-14 - exp);
      if (rshift < 32) {
        mantissa |= (1 << 23);
        mantissa = (mantissa >> rshift);
        u = (uint16_t(mantissa >> 13) & 0x3ff);
      } else {
        mantissa = 0;
        u = 0;
      }
    }

    u |= sign;

    return half_t::bitcast(u);

  #endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> bfloat16_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= bfloat16_t
template <FloatRoundStyle Round>
struct NumericConverter<float, bfloat16_t, Round> {

  using result_type = float;
  using source_type = bfloat16_t;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return static_cast<float>(s);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<bfloat16_t, float, FloatRoundStyle::round_to_nearest> {
  using result_type = bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return static_cast<bfloat16_t>(s);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<bfloat16_t, float, FloatRoundStyle::round_half_ulp_truncate> {
  using result_type = bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_truncate;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    uint32_t x32 = reinterpret_cast<uint32_t const &>(s);

    #if defined(__CUDA_ARCH__)
    if (::isfinite(s)) {
      x32 += 0x8000;
    }
    #else
    if (std::isfinite(s)) {
      x32 += 0x8000;
    }
    #endif

    uint16_t x16 = uint16_t((x32 >> 16) & 0xffff);
    return bfloat16_t::bitcast(x16);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<bfloat16_t, float, FloatRoundStyle::round_toward_zero> {
  using result_type = bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    uint32_t x32 = reinterpret_cast<uint32_t const &>(s);
    uint16_t x16 = uint16_t(x32 >> 16);

    return bfloat16_t::bitcast(x16);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> tfloat32_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= tfloat32_t
template <FloatRoundStyle Round>
struct NumericConverter<float, tfloat32_t, Round> {

  using result_type = float;
  using source_type = tfloat32_t;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return static_cast<float>(s);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<tfloat32_t, float, FloatRoundStyle::round_to_nearest> {
  using result_type = tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    unsigned storage = reinterpret_cast<unsigned const &>(s);

    if ((storage & 0x7f800000) != 0x7f800000) {

      bool mantissa_bit = ((storage & (1 << 13)) != 0);
      bool round_bit = ((storage & (1 << 12)) != 0);
      bool sticky_bit = ((storage & ((1 << 12) - 1)) != 0);

      if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
        storage += uint32_t(1 << 13);
      }

      // Note, the following is intentionally commented out. TF32
      // does not define the low order bits, so they may be left in
      // an undefined state. 
      //
      // By not truncating these bit explicitly, we avoid an extra logical
      // operation.
      //
      // TF32 may be implicitly converted to float by performing this
      // operation as needed.
      //
      // storage = (storage & ~0x1fff);
    }
    else if (storage & ~0xff800000) {
      storage = 0x7fffffff;
    }

    return tfloat32_t::bitcast(storage);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<tfloat32_t, float, FloatRoundStyle::round_half_ulp_truncate> {
  using result_type = tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_truncate;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return tfloat32_t::round_half_ulp_truncate(s);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// This rounding operation is similar to half_ulp_truncate except it rounds denorms toward zero.
/// It avoids predicated code, though it requires a temporary register.
template <>
struct NumericConverter<tfloat32_t, float, FloatRoundStyle::round_half_ulp_trunc_dntz> {
  using result_type = tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_trunc_dntz;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    unsigned y = reinterpret_cast<unsigned const &>(s);
    y = y & 0xff800000;
    float d = reinterpret_cast<float const &>(y);
    float z = d / float(1 << 11) + s;

    return reinterpret_cast<result_type const &>(z);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <>
struct NumericConverter<tfloat32_t, float, FloatRoundStyle::round_toward_zero> {
  using result_type = tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    uint32_t x = reinterpret_cast<uint32_t const &>(s);
    return tfloat32_t::bitcast(x & 0xffffe000);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion operator for float to tfloat32_t big and small values
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  FloatRoundStyle RoundBig = FloatRoundStyle::round_toward_zero,
  FloatRoundStyle RoundSmall = FloatRoundStyle::round_half_ulp_truncate
>
struct NumericConverterFastF32 {

  // result_type holds big tfloat32_t at idx(0) and small tfloat32_t at idx(1)
  using result_type = Array<tfloat32_t, 2>; 

  // source data type
  using source_type = float;

  // rounding styles for big and small part
  static FloatRoundStyle const kRoundBig = RoundBig;
  static FloatRoundStyle const kRoundSmall = RoundSmall;

  CUTLASS_HOST_DEVICE
    static result_type convert(source_type const & source) {

    result_type result;
    NumericConverter<tfloat32_t, float, kRoundBig> convert_big_;
    NumericConverter<tfloat32_t, float, kRoundSmall> convert_small_;

    // convert and fill tfloat32_t big at idx 0
    result[0] = convert_big_(source);

    // convert and fill tfloat32_t small at idx 1
    result[1] = convert_small_(source - static_cast<float>(result[0]));

    return result;
  }

  CUTLASS_HOST_DEVICE
    result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion and Clamp operator for Integers
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S
>
struct NumericConverterClamp {

  using result_type = T;
  using source_type = S;

  CUTLASS_HOST_DEVICE
    static result_type convert(source_type const & s) {
    NumericConverter<result_type, source_type> convert_op;
    result_type const kClamp_max = platform::numeric_limits<result_type>::max();
    result_type const kClamp_min = platform::numeric_limits<result_type>::lowest();
    if (s < (source_type)kClamp_min) 
      return kClamp_min;
    if (s > (source_type)kClamp_max)
      return kClamp_max;
    return convert_op(s);
  }

  CUTLASS_HOST_DEVICE
    result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion operator for Array
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Conversion operator for Array
template <
  typename T,
  typename S,
  int N,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename Transform = cutlass::transform::thread::UnaryTransform::Identity
>
struct NumericArrayConverter {

  using result_type = Array<T, N>;
  using source_type = Array<S, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value ||
                platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result;
    NumericConverter<T, S, Round> convert_;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      if( platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value )
      {
        result[i] = convert_(s[i]);
      } else { // conjugate
        result[i] = conj(convert_(s[i]));
      }
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

template <
  typename T,
  int N,
  FloatRoundStyle Round,
  typename Transform
>
struct NumericArrayConverter<T, T, N, Round, Transform> {

  using result_type = Array<T, N>;
  using source_type = Array<T, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value ||
                platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
      if( platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value )
      {
          return s;
      } else {
          result_type result;
          for (int i = 0; i < N; ++i) {
              result[i] = conj(s[i]);
          }
          return result;
      }
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<half, 2> <= Array<float, 2>, round to nearest
template <>
struct NumericArrayConverter<half_t, float, 2, FloatRoundStyle::round_to_nearest> {

  using result_type = Array<half_t, 2>;
  using source_type = Array<float, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    Array<half_t, 2> result;

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
      reinterpret_cast<__half2 &>(result) = __float22half2_rn(reinterpret_cast<float2 const &>(source));
    #else
      NumericConverter<half_t, float, round_style> convert_;
      result[0] = convert_(source[0]);
      result[1] = convert_(source[1]);
    #endif
    
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for Array<float, 2> <= Array<half_t, 2>, round to nearest
template <FloatRoundStyle Round>
struct NumericArrayConverter<float, half_t, 2, Round> {

  using result_type = Array<float, 2>;
  using source_type = Array<half_t, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    Array<float, 2> result;

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
      reinterpret_cast<float2 &>(result) = __half22float2(reinterpret_cast<__half2 const &>(source));
    #else
      NumericConverter<float, half_t, round_style> convert_;
      result[0] = convert_(source[0]);
      result[1] = convert_(source[1]);
    #endif
    
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<half> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<half_t, float, N, Round> {

  using result_type = Array<half_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<half_t, float, 2, Round> convert_vector_;
    NumericConverter<half_t, float, Round> convert_element_;

    result_type result;

    Array<half_t, 2> *result_ptr = reinterpret_cast<Array<half_t, 2> *>(&result);
    Array<float, 2> const *source_ptr = reinterpret_cast<Array<float, 2> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};


/// Partial specialization for Array<half> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, half_t, N, Round> {

  using result_type = Array<float, N>;
  using source_type = Array<half_t, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<float, half_t, 2, Round> convert_vector_;
    NumericConverter<float, half_t, Round> convert_element_;

    result_type result;

    Array<float, 2> *result_ptr = reinterpret_cast<Array<float, 2> *>(&result);
    Array<half_t, 2> const *source_ptr = reinterpret_cast<Array<half_t, 2> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<bfloat16_t, 2> <= Array<float, 2>, round to nearest
template <>
struct NumericArrayConverter<bfloat16_t, float, 2, FloatRoundStyle::round_to_nearest> {

  using result_type = Array<bfloat16_t, 2>;
  using source_type = Array<float, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    unsigned d;

    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(source[1]), "f"(source[0]) );

    return reinterpret_cast<result_type const &>(d);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<half> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<bfloat16_t, float, N, Round> {

  using result_type = Array<bfloat16_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<bfloat16_t, float, 2, Round> convert_vector_;
    NumericConverter<bfloat16_t, float, Round> convert_element_;

    result_type result;

    Array<bfloat16_t, 2> *result_ptr = reinterpret_cast<Array<bfloat16_t, 2> *>(&result);
    Array<float, 2> const *source_ptr = reinterpret_cast<Array<float, 2> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#endif // if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

/////////////////////////////////////////////////////////////////////////////////////////////////

// Conditional guards to enable partial specialization for packed integers 
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 720) && \
    ((__CUDACC_VER_MAJOR__ > 10) ||                     \
     ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2)))

/// Partial specialization for Array<int8_t, 1> <= Array<int, 1>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, 1, Round> {

  using result_type = Array<int8_t, 1>;
  using source_type = Array<int, 1>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericConverter<int8_t, int, Round> convert_element_;

    result_type result;

    result[0] = convert_element_(source[0]);
   
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for Array<int8_t, 2> <= Array<int, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, 2, Round> {

  using result_type = Array<int8_t, 2>;
  using source_type = Array<int, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    uint32_t tmp;

    asm volatile(
      "cvt.pack.sat.s8.s32.b32   %0, %2, %1, 0;\n"
      : "=r"(tmp) : "r"(source[0]), "r"(source[1]));

    uint16_t out = (tmp & 0xffff);
    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for Array<int8_t, 4> <= Array<int, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, 4, Round> {

  using result_type = Array<int8_t, 4>;
  using source_type = Array<int, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
      "{ .reg .u32 r4;"
      "cvt.pack.sat.s8.s32.b32   r4, %4, %3, 0;"
      "cvt.pack.sat.s8.s32.b32   %0, %2, %1, r4;"
      "}"
      : "=r"(out) : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for Array<int8_t> <= Array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, N, Round> {
  static_assert(!(N % 4), "N must be multiple of 4.");

  using result_type = Array<int8_t, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<int8_t, int, 4, Round> convert_vector_;

    result_type result;

    Array<int8_t, 4> *result_ptr = reinterpret_cast<Array<int8_t, 4> *>(&result);
    Array<int, 4> const *source_ptr = reinterpret_cast<Array<int, 4> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for Array<uint8_t, 1> <= Array<int, 1>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, 1, Round> {

  using result_type = Array<uint8_t, 1>;
  using source_type = Array<int, 1>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericConverter<uint8_t, int, Round> convert_element_;

    result_type result;

    result[0] = convert_element_(source[0]);
   
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for Array<uint8_t, 2> <= Array<int, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, 2, Round> {

  using result_type = Array<uint8_t, 2>;
  using source_type = Array<int, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    uint32_t tmp;

    asm volatile(
      "cvt.pack.sat.u8.s32.b32   %0, %2, %1, 0;\n"
      : "=r"(tmp) : "r"(source[0]), "r"(source[1]));

    uint16_t out = (tmp & 0xffff);
    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for Array<uint8_t, 4> <= Array<int, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, 4, Round> {

  using result_type = Array<uint8_t, 4>;
  using source_type = Array<int, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
      "{ .reg .u32 r4;"
      "cvt.pack.sat.u8.s32.b32   r4, %4, %3, 0;"
      "cvt.pack.sat.u8.s32.b32   %0, %2, %1, r4;"
      "}"
      : "=r"(out) : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for Array<int8_t> <= Array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, N, Round> {
  static_assert(!(N % 4), "N must be multiple of 4.");

  using result_type = Array<uint8_t, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<uint8_t, int, 4, Round> convert_vector_;

    result_type result;

    Array<uint8_t, 4> *result_ptr = reinterpret_cast<Array<uint8_t, 4> *>(&result);
    Array<int, 4> const *source_ptr = reinterpret_cast<Array<int, 4> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && \
    ((__CUDACC_VER_MAJOR__ > 10) ||                     \
     ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2)))

/// Partial specialization for Array<int4b_t, 8> <= Array<int, 8>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int4b_t, int, 8, Round> {

  using result_type = Array<int4b_t, 8>;
  using source_type = Array<int, 8>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
        "{ .reg .u32 r4;"
        "cvt.pack.sat.s4.s32.b32   r4, %8, %7, 0;"
        "cvt.pack.sat.s4.s32.b32   r4, %6, %5, r4;"
        "cvt.pack.sat.s4.s32.b32   r4, %4, %3, r4;"
        "cvt.pack.sat.s4.s32.b32   %0, %2, %1, r4;"
        "}"
        : "=r"(out)
        : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]),
          "r"(source[4]), "r"(source[5]), "r"(source[6]), "r"(source[7]));

    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for Array<int4b_t> <= Array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int4b_t, int, N, Round> {
  static_assert(!(N % 8), "N must be multiple of 8.");

  using result_type = Array<int4b_t, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<int4b_t, int, 8, Round> convert_vector_;

    result_type result;

    Array<int4b_t, 8> *result_ptr = reinterpret_cast<Array<int4b_t, 8> *>(&result);
    Array<int, 8> const *source_ptr = reinterpret_cast<Array<int, 8> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for Array<uint4b_t, 8> <= Array<int, 8>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint4b_t, int, 8, Round> {

  using result_type = Array<uint4b_t, 8>;
  using source_type = Array<int, 8>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
        "{ .reg .u32 r4;"
        "cvt.pack.sat.u4.s32.b32   r4, %8, %7, 0;"
        "cvt.pack.sat.u4.s32.b32   r4, %6, %5, r4;"
        "cvt.pack.sat.u4.s32.b32   r4, %4, %3, r4;"
        "cvt.pack.sat.u4.s32.b32   %0, %2, %1, r4;"
        "}"
        : "=r"(out)
        : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]),
          "r"(source[4]), "r"(source[5]), "r"(source[6]), "r"(source[7]));

    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/// Partial specialization for Array<int4b_t> <= Array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint4b_t, int, N, Round> {
  static_assert(!(N % 8), "N must be multiple of 8.");

  using result_type = Array<uint4b_t, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<uint4b_t, int, 8, Round> convert_vector_;

    result_type result;

    Array<uint4b_t, 8> *result_ptr = reinterpret_cast<Array<uint4b_t, 8> *>(&result);
    Array<int, 8> const *source_ptr = reinterpret_cast<Array<int, 8> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

#endif  // Conditional guards to enable partial specialization for packed integers

/////////////////////////////////////////////////////////////////////////////////////////////////

/// FastNumericArrayConverter only works when the source is within center range.
/// Conversion operator for Array.  See the comments before
/// FastLinearCombinationClamp.
template <typename T, typename S, int N,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
struct FastNumericArrayConverter {
  using result_type = Array<T, N>;
  using source_type = Array<S, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const &s) {
    result_type result;
    NumericArrayConverter<T, S, N, Round> convert_;

    return convert_(s);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) { return convert(s); }
};

/// Partial specialization for Array<float> <= Array<int>
template <typename T, int N, FloatRoundStyle Round>
struct FastNumericArrayConverter<float, T, N, Round> {
  using result_type = Array<float, N>;
  using source_type = Array<T, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int tmp = source[i] + 1262485504 /*0x4B400000*/;
      result[i] = reinterpret_cast<float const &>(tmp) - 12582912.0f;
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) { return convert(s); }
};

/// Partial specialization for Array<int8_t, 4> <= Array<float, 4>
template <FloatRoundStyle Round>
struct FastNumericArrayConverter<int8_t, float, 4, Round> {
  using result_type = Array<int8_t, 4>;
  using source_type = Array<float, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    Array<int32_t, 4> result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      float tmp = source[i] + 12582912.0f;
      result[i] = reinterpret_cast<int32_t const &>(tmp);
    }

    result[0] = __byte_perm(result[0], result[1], 0x40);
    result[2] = __byte_perm(result[2], result[3], 0x40);
    result[0] = __byte_perm(result[0], result[2], 0x5410);

    return reinterpret_cast<result_type const &>(result[0]);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) { return convert(s); }
};

/// Partial specialization for Array<int8_t> <= Array<float>
template <int N, FloatRoundStyle Round>
struct FastNumericArrayConverter<int8_t, float, N, Round> {
  static_assert(!(N % 4), "N must be multiple of 4.");

  using result_type = Array<int8_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    FastNumericArrayConverter<int8_t, float, 4, Round> convert_vector_;

    result_type result;

    Array<int8_t, 4> *result_ptr =
        reinterpret_cast<Array<int8_t, 4> *>(&result);
    Array<float, 4> const *source_ptr =
        reinterpret_cast<Array<float, 4> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) { return convert(s); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines preferred rounding mode for a pair of types
template <typename T, typename S>
struct PreferredRoundingMode {
  static FloatRoundStyle const kRound = FloatRoundStyle::round_to_nearest;
};

/// Defines preferred rounding mode for a pair of types
template <>
struct PreferredRoundingMode<tfloat32_t, float> {
  static FloatRoundStyle const kRound = FloatRoundStyle::round_half_ulp_truncate;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Packs predicates into an array.
template <int N>
struct PackPredicates {
  using result_type = Array<uint1b_t, N>;

  static_assert(!(N % 4), "Must pack predicates in a count that is a multiple of 4");

  CUTLASS_HOST_DEVICE
  result_type operator()(bool const predicates[]) {

    result_type packed;
    packed.clear();

    int const kWordSize = 8;
    uint8_t *bytes = reinterpret_cast<uint8_t *>(packed.data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int word_idx = (i / kWordSize);
      int bit_idx = (i % kWordSize);

      uint8_t mask = ((predicates[i] ? 1u : 0u) << bit_idx);
      bytes[word_idx] = (bytes[word_idx] | mask);
    }
    return packed;
  }
};

/// Packs predicates into an array
template <int N>
struct UnpackPredicates {
  using result_type = Array<uint1b_t, N>;

  static_assert(!(N % 4), "Must unpack predicates in a count that is a multiple of 4");

  CUTLASS_HOST_DEVICE
  void operator()(bool predicates[], result_type const &packed) {

    int const kWordSize = 8;
    uint8_t const *bytes = reinterpret_cast<uint8_t const *>(packed.data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int word_idx = (i / kWordSize);
      int bit_idx = (i % kWordSize);

      predicates[i] = bool((bytes[word_idx] >> bit_idx) & 0x1);
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
