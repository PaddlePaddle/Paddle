// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <c10/util/BFloat16.h>
// #include <c10/util/Float4_e2m1fn_x2.h>
#include <c10/util/Float8_e4m3fn.h>
// #include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
// #include <c10/util/Float8_e5m2fnuz.h>
// #include <c10/util/Float8_e8m0fnu.h>
#include <c10/util/Half.h>
// #include <c10/util/bits.h>
#include <c10/util/complex.h>
// #include <c10/util/qint32.h>
// #include <c10/util/qint8.h>
// #include <c10/util/quint2x4.h>
// #include <c10/util/quint4x2.h>
// #include <c10/util/quint8.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "paddle/common/macros.h"

namespace c10 {

// dummy struct for uint1 to uint7, actual functionality
// of these dtypes will be implemented in python with Tensor subclass
template <unsigned int N>
struct dummy_uint1_7_t {};

// dummy struct for int1 to int7, actual functionality
// of these dtypes will be implemented in python with Tensor subclass
template <unsigned int N>
struct dummy_int1_7_t {};

#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_)       \
  _(uint8_t, UINT8, Byte)  /* 0 */                             \
  _(int8_t, INT8, Char)    /* 1 */                             \
  _(int16_t, INT16, Short) /* 2 */                             \
  _(int, INT32, Int)       /* 3 */                             \
  _(int64_t, INT64, Long)  /* 4 */                             \
  _(at::Half, FLOAT16, Half)                                   \
  _(float, FLOAT32, Float)                            /* 6 */  \
  _(double, FLOAT64, Double)                          /* 7 */  \
  _(c10::complex<float>, COMPLEX64, ComplexFloat)     /* 9 */  \
  _(c10::complex<double>, COMPLEX128, ComplexDouble)  /* 10 */ \
  _(bool, BOOL, Bool)                                 /* 11 */ \
  _(at::BFloat16, BFLOAT16, BFloat16)                 /* 15 */ \
  _(c10::Float8_e5m2, FLOAT8_E5M2, Float8_e5m2)       /* 23 */ \
  _(c10::Float8_e4m3fn, FLOAT8_E4M3FN, Float8_e4m3fn) /* 24 */ \
  _(uint16_t, UINT16, UInt16)                         /* 27 */ \
  _(uint32_t, UINT32, UInt32)                         /* 28 */ \
  _(uint64_t, UINT64, UInt64)                         /* 29 */ \
  _(c10::dummy_uint1_7_t<1>, UInt1, UInt1)            /* 30 */ \
  _(c10::dummy_uint1_7_t<2>, UInt2, UInt2)            /* 31 */ \
  _(c10::dummy_uint1_7_t<3>, UInt3, UInt3)            /* 32 */ \
  _(c10::dummy_uint1_7_t<4>, UInt4, UInt4)            /* 33 */ \
  _(c10::dummy_uint1_7_t<5>, UInt5, UInt5)            /* 34 */ \
  _(c10::dummy_uint1_7_t<6>, UInt6, UInt6)            /* 35 */ \
  _(c10::dummy_uint1_7_t<7>, UInt7, UInt7)            /* 36 */ \
  _(c10::dummy_int1_7_t<1>, Int1, Int1)               /* 37 */ \
  _(c10::dummy_int1_7_t<2>, Int2, Int2)               /* 38 */ \
  _(c10::dummy_int1_7_t<3>, Int3, Int3)               /* 39 */ \
  _(c10::dummy_int1_7_t<4>, Int4, Int4)               /* 40 */ \
  _(c10::dummy_int1_7_t<5>, Int5, Int5)               /* 41 */ \
  _(c10::dummy_int1_7_t<6>, Int6, Int6)               /* 42 */ \
  _(c10::dummy_int1_7_t<7>, Int7, Int7)               /* 43 */

#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(_) \
  _(uint8_t, Byte)                             \
  _(int8_t, Char)                              \
  _(int16_t, Short)                            \
  _(int, Int)                                  \
  _(int64_t, Long)                             \
  _(float, Float)                              \
  _(double, Double)                            \
  _(c10::complex<float>, ComplexFloat)         \
  _(c10::complex<double>, ComplexDouble)       \
  _(bool, Bool)                                \
  _(at::BFloat16, BFloat16)                    \
  _(c10::Float8_e5m2, Float8_e5m2)              \
  _(c10::Float8_e4m3fn, Float8_e4m3fn)

#define AT_FORALL_QINT_TYPES(_) \
  _(c10::qint8, QInt8)          \
  _(c10::quint8, QUInt8)        \
  _(c10::qint32, QInt32)        \
  _(c10::quint4x2, QUInt4x2)    \
  _(c10::quint2x4, QUInt2x4)

#define FOREACH_PADDLE_AND_TORCH_DTYPES(_)            \
  _(uint8_t, UINT8, Byte)                             \
  _(int8_t, INT8, Char)                               \
  _(int16_t, INT16, Short)                            \
  _(int32_t, INT32, Int)                              \
  _(int64_t, INT64, Long)                             \
  _(at::Half, FLOAT16, Half)                          \
  _(float, FLOAT32, Float)                            \
  _(double, FLOAT64, Double)                          \
  _(c10::complex<float>, COMPLEX64, ComplexFloat)     \
  _(c10::complex<double>, COMPLEX128, ComplexDouble)  \
  _(bool, BOOL, Bool)                                 \
  _(at::BFloat16, BFLOAT16, BFloat16)                 \
  _(c10::Float8_e5m2, FLOAT8_E5M2, Float8_e5m2)       \
  _(c10::Float8_e4m3fn, FLOAT8_E4M3FN, Float8_e4m3fn) \
  _(uint16_t, UINT16, UInt16)                         \
  _(uint32_t, UINT32, UInt32)

enum class PADDLE_API ScalarType : int8_t {
#define DEFINE_ST_ENUM_VAL_(_1, _2, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ST_ENUM_VAL_)
#undef DEFINE_ENUM_ST_ENUM_VAL_
      Undefined,
  NumOptions
};

#define DEFINE_CONSTANT(_1, _2, name) \
  constexpr ScalarType k##name = ScalarType::name;

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

inline const char* toString(ScalarType t) {
#define DEFINE_CASE(_1, _2, name) \
  case ScalarType::name:          \
    return #name;

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE
}

inline size_t elementSize(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, _2, name) \
  case ScalarType::name:                       \
    return sizeof(ctype);

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CASE_ELEMENTSIZE_CASE)
    default:
      TORCH_CHECK(false, "Unknown ScalarType");
  }
#undef CASE_ELEMENTSIZE_CASE
}

inline bool isIntegralType(ScalarType t, bool includeBool) {
  bool isIntegral = (t == ScalarType::Byte || t == ScalarType::Char ||
                     t == ScalarType::Int || t == ScalarType::Long ||
                     t == ScalarType::Short || t == ScalarType::UInt16 ||
                     t == ScalarType::UInt32 || t == ScalarType::UInt64);

  return isIntegral || (includeBool && t == ScalarType::Bool);
}

inline bool isFloat8Type(ScalarType t) {
  return t == ScalarType::Float8_e5m2 || t == ScalarType::Float8_e4m3fn;
  //  ||  t == ScalarType::Float8_e5m2fnuz
  //   ||  t == ScalarType::Float8_e4m3fnuz
  //   || t == ScalarType::Float8_e8m0fnu
}

inline bool isReducedFloatingType(ScalarType t) {
  return t == ScalarType::Half || t == ScalarType::BFloat16 || isFloat8Type(t);
  //||  t == ScalarType::Float4_e2m1fn_x2
}

inline bool isFloatingType(ScalarType t) {
  return t == ScalarType::Double || t == ScalarType::Float ||
         isReducedFloatingType(t);
}

inline bool isComplexType(ScalarType t) {
  return (
      /* t == ScalarType::ComplexHalf || */ t == ScalarType::ComplexFloat ||
      t == ScalarType::ComplexDouble);
}

// inline bool isQIntType(ScalarType t) {
//   // Don't forget to extend this when adding new QInt types
//   return t == ScalarType::QInt8 || t == ScalarType::QUInt8 ||
//          t == ScalarType::QInt32 || t == ScalarType::QUInt4x2 ||
//          t == ScalarType::QUInt2x4;
// }

// inline ScalarType toUnderlying(ScalarType t) {
//   switch (t) {
//     case ScalarType::QUInt8:
//     case ScalarType::QUInt4x2:
//       [[fallthrough]];
//     case ScalarType::QUInt2x4:
//       return ScalarType::Byte;
//     case ScalarType::QInt8:
//       return ScalarType::Char;
//     case ScalarType::QInt32:
//       return ScalarType::Int;
//     default:
//       return t;
//   }
// }

}  // namespace c10

namespace torch {
using c10::ScalarType;
}  // namespace torch
