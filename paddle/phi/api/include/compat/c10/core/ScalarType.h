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

// If you modified ScalarType foundational definitions in this file, please
// also sync your changes into torch/headeronly/core/ScalarType.h.
#include <torch/headeronly/core/ScalarType.h>

#include <c10/util/Exception.h>

#include <limits>
#include <sstream>

namespace c10 {

#define DEFINE_CONSTANT(_1, _2, name) \
  constexpr ScalarType k##name = ScalarType::name;

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

constexpr ScalarType kUndefined = ScalarType::Undefined;

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

}  // namespace c10

namespace at {
using c10::CppTypeToScalarType;
using c10::ScalarType;
}  // namespace at
namespace torch {
using c10::CppTypeToScalarType;
using c10::ScalarType;
}  // namespace torch
