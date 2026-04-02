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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <sstream>

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
  _(at::Float8_e5m2, Float8_e5m2)

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

// These are used to map ScalarTypes to C++ types.

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

#define DEFINE_CONSTANT(_1, _2, name) \
  constexpr ScalarType k##name = ScalarType::name;

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

constexpr ScalarType kComplexHalf = ScalarType::ComplexHalf;
constexpr ScalarType kQInt8 = ScalarType::QInt8;
constexpr ScalarType kQUInt8 = ScalarType::QUInt8;
constexpr ScalarType kQInt32 = ScalarType::QInt32;
constexpr ScalarType kQUInt4x2 = ScalarType::QUInt4x2;
constexpr ScalarType kQUInt2x4 = ScalarType::QUInt2x4;
constexpr ScalarType kBits1x8 = ScalarType::Bits1x8;
constexpr ScalarType kBits2x4 = ScalarType::Bits2x4;
constexpr ScalarType kBits4x2 = ScalarType::Bits4x2;
constexpr ScalarType kBits8 = ScalarType::Bits8;
constexpr ScalarType kBits16 = ScalarType::Bits16;
constexpr ScalarType kFloat8_e5m2fnuz = ScalarType::Float8_e5m2fnuz;
constexpr ScalarType kFloat8_e4m3fnuz = ScalarType::Float8_e4m3fnuz;
constexpr ScalarType kFloat8_e8m0fnu = ScalarType::Float8_e8m0fnu;
constexpr ScalarType kFloat4_e2m1fn_x2 = ScalarType::Float4_e2m1fn_x2;
constexpr ScalarType kUndefined = ScalarType::Undefined;

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
    case ScalarType::ComplexHalf:
      return "ComplexHalf";
    case ScalarType::QInt8:
      return "QInt8";
    case ScalarType::QUInt8:
      return "QUInt8";
    case ScalarType::QInt32:
      return "QInt32";
    case ScalarType::QUInt4x2:
      return "QUInt4x2";
    case ScalarType::QUInt2x4:
      return "QUInt2x4";
    case ScalarType::Bits1x8:
      return "Bits1x8";
    case ScalarType::Bits2x4:
      return "Bits2x4";
    case ScalarType::Bits4x2:
      return "Bits4x2";
    case ScalarType::Bits8:
      return "Bits8";
    case ScalarType::Bits16:
      return "Bits16";
    case ScalarType::Float8_e5m2fnuz:
      return "Float8_e5m2fnuz";
    case ScalarType::Float8_e4m3fnuz:
      return "Float8_e4m3fnuz";
    case ScalarType::Float8_e8m0fnu:
      return "Float8_e8m0fnu";
    case ScalarType::Float4_e2m1fn_x2:
      return "Float4_e2m1fn_x2";
    case ScalarType::Undefined:
      return "Undefined";
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
    case ScalarType::ComplexHalf:
      return sizeof(at::Half) * 2;
    case ScalarType::QInt8:
    case ScalarType::QUInt8:
    case ScalarType::QUInt4x2:
    case ScalarType::QUInt2x4:
    case ScalarType::Bits1x8:
    case ScalarType::Bits2x4:
    case ScalarType::Bits4x2:
    case ScalarType::Bits8:
    case ScalarType::Float8_e5m2fnuz:
    case ScalarType::Float8_e4m3fnuz:
    case ScalarType::Float8_e8m0fnu:
    case ScalarType::Float4_e2m1fn_x2:
      return 1;
    case ScalarType::QInt32:
      return 4;
    case ScalarType::Bits16:
      return 2;
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
  return t == ScalarType::Float8_e5m2 || t == ScalarType::Float8_e4m3fn ||
         t == ScalarType::Float8_e5m2fnuz || t == ScalarType::Float8_e4m3fnuz ||
         t == ScalarType::Float8_e8m0fnu;
}

inline bool isReducedFloatingType(ScalarType t) {
  return t == ScalarType::Half || t == ScalarType::BFloat16 ||
         isFloat8Type(t) || t == ScalarType::Float4_e2m1fn_x2;
}

inline bool isFloatingType(ScalarType t) {
  return t == ScalarType::Double || t == ScalarType::Float ||
         isReducedFloatingType(t);
}

inline bool isComplexType(ScalarType t) {
  return (t == ScalarType::ComplexHalf || t == ScalarType::ComplexFloat ||
          t == ScalarType::ComplexDouble);
}

inline bool isSignedType(ScalarType t) {
#define CASE_ISSIGNED(name)     \
  case ScalarType::name:        \
    return std::numeric_limits< \
        ::c10::impl::ScalarTypeToCPPTypeT<ScalarType::name>>::is_signed;

  switch (t) {
    // Signed integer types (using numeric_limits)
    CASE_ISSIGNED(Char);   // int8_t
    CASE_ISSIGNED(Short);  // int16_t
    CASE_ISSIGNED(Int);    // int32_t
    CASE_ISSIGNED(Long);   // int64_t

    // Signed integer types (dummy types, explicitly return true)
    case ScalarType::Int1:
    case ScalarType::Int2:
    case ScalarType::Int3:
    case ScalarType::Int4:
    case ScalarType::Int5:
    case ScalarType::Int6:
    case ScalarType::Int7:
      return true;

      // Signed floating point types (using numeric_limits)
      CASE_ISSIGNED(Half);    // float16
      CASE_ISSIGNED(Float);   // float32
      CASE_ISSIGNED(Double);  // float64
      CASE_ISSIGNED(BFloat16);
      CASE_ISSIGNED(Float8_e5m2);
      CASE_ISSIGNED(Float8_e4m3fn);

    // Complex types (treated as signed)
    case ScalarType::ComplexHalf:
    case ScalarType::ComplexFloat:
    case ScalarType::ComplexDouble:
      return true;

    // Signed quantized types (explicitly return true)
    case ScalarType::QInt8:
    case ScalarType::QInt32:
      return true;

      // Unsigned integer types (using numeric_limits)
      CASE_ISSIGNED(Byte);  // uint8_t

    // Unsigned integer types (explicitly return false)
    case ScalarType::UInt16:
    case ScalarType::UInt32:
    case ScalarType::UInt64:
    case ScalarType::UInt1:
    case ScalarType::UInt2:
    case ScalarType::UInt3:
    case ScalarType::UInt4:
    case ScalarType::UInt5:
    case ScalarType::UInt6:
    case ScalarType::UInt7:
      return false;

    // Unsigned quantized types (explicitly return false)
    case ScalarType::QUInt8:
    case ScalarType::QUInt4x2:
    case ScalarType::QUInt2x4:
    case ScalarType::Bits1x8:
    case ScalarType::Bits2x4:
    case ScalarType::Bits4x2:
    case ScalarType::Bits8:
    case ScalarType::Bits16:
      return false;

      // Bool is unsigned (using numeric_limits)
      CASE_ISSIGNED(Bool);

    case ScalarType::Float8_e5m2fnuz:
    case ScalarType::Float8_e4m3fnuz:
    case ScalarType::Float8_e8m0fnu:
    case ScalarType::Float4_e2m1fn_x2:
      return true;

    // Invalid/undefined types - should not happen in normal usage
    // If this is hit, it indicates a programming error or unsupported type
    case ScalarType::Undefined:
    case ScalarType::NumOptions: {
      std::ostringstream oss;
      oss << "isSignedType: Invalid or unsupported ScalarType value: "
          << toString(t) << " (" << static_cast<int>(t) << ")";
      PD_THROW(oss.str());
      return false;  // Unreachable, but satisfies compiler
    }

      // Note: If a new ScalarType is added to the enum but not handled here,
      // the compiler will warn about missing case. This ensures all types are
      // explicitly handled.
  }
#undef CASE_ISSIGNED
  return false;  // Unreachable, but satisfies compiler
}

inline std::ostream& operator<<(std::ostream& stream, ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

}  // namespace c10

namespace at {
using c10::CppTypeToScalarType;
using c10::ScalarType;
}  // namespace at
namespace torch {
using c10::CppTypeToScalarType;
using c10::ScalarType;
}  // namespace torch
