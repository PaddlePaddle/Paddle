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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Float4_e2m1fn_x2.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Float8_e8m0fnu.h>
#include <c10/util/Half.h>
#include <c10/util/bits.h>
#include <c10/util/complex.h>
#include <c10/util/qint32.h>
#include <c10/util/qint8.h>
#include <c10/util/quint2x4.h>
#include <c10/util/quint4x2.h>
#include <c10/util/quint8.h>

#include <cstdint>
#include <ostream>
#include <type_traits>

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

#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_)                \
  _(uint8_t, UINT8, Byte)                                      /* 0 */  \
  _(int8_t, INT8, Char)                                        /* 1 */  \
  _(int16_t, INT16, Short)                                     /* 2 */  \
  _(int, INT32, Int)                                           /* 3 */  \
  _(int64_t, INT64, Long)                                      /* 4 */  \
  _(at::Half, FLOAT16, Half)                                   /* 5 */  \
  _(float, FLOAT32, Float)                                     /* 6 */  \
  _(double, FLOAT64, Double)                                   /* 7 */  \
  _(c10::complex<at::Half>, ComplexHalf, ComplexHalf)          /* 8 */  \
  _(c10::complex<float>, COMPLEX64, ComplexFloat)              /* 9 */  \
  _(c10::complex<double>, COMPLEX128, ComplexDouble)           /* 10 */ \
  _(bool, BOOL, Bool)                                          /* 11 */ \
  _(c10::qint8, QInt8, QInt8)                                  /* 12 */ \
  _(c10::quint8, QUInt8, QUInt8)                               /* 13 */ \
  _(c10::qint32, QInt32, QInt32)                               /* 14 */ \
  _(at::BFloat16, BFLOAT16, BFloat16)                          /* 15 */ \
  _(c10::quint4x2, QUInt4x2, QUInt4x2)                         /* 16 */ \
  _(c10::quint2x4, QUInt2x4, QUInt2x4)                         /* 17 */ \
  _(c10::bits1x8, Bits1x8, Bits1x8)                            /* 18 */ \
  _(c10::bits2x4, Bits2x4, Bits2x4)                            /* 19 */ \
  _(c10::bits4x2, Bits4x2, Bits4x2)                            /* 20 */ \
  _(c10::bits8, Bits8, Bits8)                                  /* 21 */ \
  _(c10::bits16, Bits16, Bits16)                               /* 22 */ \
  _(c10::Float8_e5m2, FLOAT8_E5M2, Float8_e5m2)                /* 23 */ \
  _(c10::Float8_e4m3fn, FLOAT8_E4M3FN, Float8_e4m3fn)          /* 24 */ \
  _(c10::Float8_e5m2fnuz, Float8_e5m2fnuz, Float8_e5m2fnuz)    /* 25 */ \
  _(c10::Float8_e4m3fnuz, Float8_e4m3fnuz, Float8_e4m3fnuz)    /* 26 */ \
  _(uint16_t, UINT16, UInt16)                                  /* 27 */ \
  _(uint32_t, UINT32, UInt32)                                  /* 28 */ \
  _(uint64_t, UINT64, UInt64)                                  /* 29 */ \
  _(c10::dummy_uint1_7_t<1>, UInt1, UInt1)                     /* 30 */ \
  _(c10::dummy_uint1_7_t<2>, UInt2, UInt2)                     /* 31 */ \
  _(c10::dummy_uint1_7_t<3>, UInt3, UInt3)                     /* 32 */ \
  _(c10::dummy_uint1_7_t<4>, UInt4, UInt4)                     /* 33 */ \
  _(c10::dummy_uint1_7_t<5>, UInt5, UInt5)                     /* 34 */ \
  _(c10::dummy_uint1_7_t<6>, UInt6, UInt6)                     /* 35 */ \
  _(c10::dummy_uint1_7_t<7>, UInt7, UInt7)                     /* 36 */ \
  _(c10::dummy_int1_7_t<1>, Int1, Int1)                        /* 37 */ \
  _(c10::dummy_int1_7_t<2>, Int2, Int2)                        /* 38 */ \
  _(c10::dummy_int1_7_t<3>, Int3, Int3)                        /* 39 */ \
  _(c10::dummy_int1_7_t<4>, Int4, Int4)                        /* 40 */ \
  _(c10::dummy_int1_7_t<5>, Int5, Int5)                        /* 41 */ \
  _(c10::dummy_int1_7_t<6>, Int6, Int6)                        /* 42 */ \
  _(c10::dummy_int1_7_t<7>, Int7, Int7)                        /* 43 */ \
  _(c10::Float8_e8m0fnu, Float8_e8m0fnu, Float8_e8m0fnu)       /* 44 */ \
  _(c10::Float4_e2m1fn_x2, Float4_e2m1fn_x2, Float4_e2m1fn_x2) /* 45 */

#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ(_) \
  _(uint8_t, Byte)                                                      \
  _(int8_t, Char)                                                       \
  _(int16_t, Short)                                                     \
  _(int, Int)                                                           \
  _(int64_t, Long)                                                      \
  _(at::Half, Half)                                                     \
  _(float, Float)                                                       \
  _(double, Double)                                                     \
  _(c10::complex<float>, ComplexFloat)                                  \
  _(c10::complex<double>, ComplexDouble)                                \
  _(bool, Bool)                                                         \
  _(at::BFloat16, BFloat16)                                             \
  _(c10::Float8_e5m2, Float8_e5m2)                                      \
  _(c10::Float8_e4m3fn, Float8_e4m3fn)

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
  _(c10::Float8_e5m2, Float8_e5m2)             \
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
  Byte = 0,
  Char = 1,
  Short = 2,
  Int = 3,
  Long = 4,
  Half = 5,
  Float = 6,
  Double = 7,
  ComplexHalf = 8,
  ComplexFloat = 9,
  ComplexDouble = 10,
  Bool = 11,
  QInt8 = 12,
  QUInt8 = 13,
  QInt32 = 14,
  BFloat16 = 15,
  QUInt4x2 = 16,
  QUInt2x4 = 17,
  Bits1x8 = 18,
  Bits2x4 = 19,
  Bits4x2 = 20,
  Bits8 = 21,
  Bits16 = 22,
  Float8_e5m2 = 23,
  Float8_e4m3fn = 24,
  Float8_e5m2fnuz = 25,
  Float8_e4m3fnuz = 26,
  UInt16 = 27,
  UInt32 = 28,
  UInt64 = 29,
  UInt1 = 30,
  UInt2 = 31,
  UInt3 = 32,
  UInt4 = 33,
  UInt5 = 34,
  UInt6 = 35,
  UInt7 = 36,
  Int1 = 37,
  Int2 = 38,
  Int3 = 39,
  Int4 = 40,
  Int5 = 41,
  Int6 = 42,
  Int7 = 43,
  Float8_e8m0fnu = 44,
  Float4_e2m1fn_x2 = 45,
  Undefined = 46,
  NumOptions = 47
};

constexpr uint16_t NumScalarTypes =
    static_cast<uint16_t>(ScalarType::NumOptions);

namespace impl {

template <c10::ScalarType N>
struct ScalarTypeToCPPType;

#define SPECIALIZE_ScalarTypeToCPPType(cpp_type, _2, scalar_type) \
  template <>                                                     \
  struct ScalarTypeToCPPType<c10::ScalarType::scalar_type> {      \
    using type = cpp_type;                                        \
                                                                  \
    static type t;                                                \
  };

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_ScalarTypeToCPPType)

#undef SPECIALIZE_ScalarTypeToCPPType

template <c10::ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;

}  // namespace impl

template <typename T>
struct CppTypeToScalarType;

#define SPECIALIZE_CppTypeToScalarType(cpp_type, _2, scalar_type) \
  template <>                                                     \
  struct CppTypeToScalarType<cpp_type>                            \
      : std::integral_constant<c10::ScalarType,                   \
                               c10::ScalarType::scalar_type> {};

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_CppTypeToScalarType)

#undef SPECIALIZE_CppTypeToScalarType

#define AT_FORALL_SCALAR_TYPES_AND(SCALARTYPE, _) \
  _(uint8_t, Byte)                                \
  _(int8_t, Char)                                 \
  _(int16_t, Short)                               \
  _(int, Int)                                     \
  _(int64_t, Long)                                \
  _(float, Float)                                 \
  _(double, Double)                               \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE>::t),  \
    SCALARTYPE)

#define AT_FORALL_SCALAR_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, _) \
  _(uint8_t, Byte)                                               \
  _(int8_t, Char)                                                \
  _(int16_t, Short)                                              \
  _(int, Int)                                                    \
  _(int64_t, Long)                                               \
  _(float, Float)                                                \
  _(double, Double)                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                   \
             ::c10::ScalarType::SCALARTYPE1>::t),                \
    SCALARTYPE1)                                                 \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                   \
             ::c10::ScalarType::SCALARTYPE2>::t),                \
    SCALARTYPE2)

#define AT_FORALL_SCALAR_TYPES_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, _) \
  _(uint8_t, Byte)                                                            \
  _(int8_t, Char)                                                             \
  _(int16_t, Short)                                                           \
  _(int, Int)                                                                 \
  _(int64_t, Long)                                                            \
  _(float, Float)                                                             \
  _(double, Double)                                                           \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE1>::t),                             \
    SCALARTYPE1)                                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE2>::t),                             \
    SCALARTYPE2)                                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE3>::t),                             \
    SCALARTYPE3)

#define AT_FORALL_COMPLEX_TYPES(_)     \
  _(c10::complex<float>, ComplexFloat) \
  _(c10::complex<double>, ComplexDouble)

inline const char* toString(ScalarType t) {
#define DEFINE_CASE(_1, _2, name) \
  case ScalarType::name:          \
    return #name;

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)
    case ScalarType::Undefined:
      return "Undefined";
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE
}

inline std::ostream& operator<<(std::ostream& stream, ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

inline bool isQIntType(ScalarType t) {
  return t == ScalarType::QInt8 || t == ScalarType::QUInt8 ||
         t == ScalarType::QInt32 || t == ScalarType::QUInt4x2 ||
         t == ScalarType::QUInt2x4;
}

inline ScalarType toUnderlying(ScalarType t) {
  switch (t) {
    case ScalarType::QUInt8:
    case ScalarType::QUInt4x2:
    case ScalarType::QUInt2x4:
      return ScalarType::Byte;
    case ScalarType::QInt8:
      return ScalarType::Char;
    case ScalarType::QInt32:
      return ScalarType::Int;
    default:
      return t;
  }
}

}  // namespace c10
