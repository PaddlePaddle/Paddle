// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <iostream>
#include <string>
#include <typeindex>
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/framework/framework.pb.h"

namespace phi {
using VarType = paddle::framework::proto::VarType;

PADDLE_API std::string VarDataTypeToString(const VarType::Type type);
TEST_API extern size_t SizeOfType(VarType::Type type);

template <typename T>
struct IsComplex : public std::false_type {};

template <typename T>
struct IsComplex<phi::dtype::complex<T>> : public std::true_type {};

template <typename T>
struct DataTypeTrait {};

// Stub handle for void
template <>
struct DataTypeTrait<void> {
  constexpr static VarType::Type DataType() { return VarType::RAW; }
};

#define _ForEachDataTypeHelper_(callback, cpp_type, proto_type) \
  callback(cpp_type, ::paddle::framework::proto::VarType::proto_type);

#define _ForEachDataType_(callback)                                           \
  _ForEachDataTypeHelper_(callback, float, FP32);                             \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::float16, FP16);             \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::bfloat16, BF16);            \
  _ForEachDataTypeHelper_(callback, double, FP64);                            \
  _ForEachDataTypeHelper_(callback, int, INT32);                              \
  _ForEachDataTypeHelper_(callback, int64_t, INT64);                          \
  _ForEachDataTypeHelper_(callback, bool, BOOL);                              \
  _ForEachDataTypeHelper_(callback, uint8_t, UINT8);                          \
  _ForEachDataTypeHelper_(callback, uint16_t, UINT16);                        \
  _ForEachDataTypeHelper_(callback, uint32_t, UINT32);                        \
  _ForEachDataTypeHelper_(callback, uint64_t, UINT64);                        \
  _ForEachDataTypeHelper_(callback, int16_t, INT16);                          \
  _ForEachDataTypeHelper_(callback, int8_t, INT8);                            \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::complex<float>, COMPLEX64); \
  _ForEachDataTypeHelper_(                                                    \
      callback, ::phi::dtype::complex<double>, COMPLEX128);                   \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::float8_e4m3fn, FP8_E4M3FN); \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::float8_e5m2, FP8_E5M2);

#define _ForEachIntDataType_(callback)                 \
  _ForEachDataTypeHelper_(callback, int, INT32);       \
  _ForEachDataTypeHelper_(callback, int64_t, INT64);   \
  _ForEachDataTypeHelper_(callback, uint8_t, UINT8);   \
  _ForEachDataTypeHelper_(callback, uint16_t, UINT16); \
  _ForEachDataTypeHelper_(callback, uint32_t, UINT32); \
  _ForEachDataTypeHelper_(callback, uint64_t, UINT64); \
  _ForEachDataTypeHelper_(callback, int16_t, INT16);   \
  _ForEachDataTypeHelper_(callback, int8_t, INT8);

#define _ForEachDataTypeSmall_(callback)                                      \
  _ForEachDataTypeHelper_(callback, float, FP32);                             \
  _ForEachDataTypeHelper_(callback, double, FP64);                            \
  _ForEachDataTypeHelper_(callback, int, INT32);                              \
  _ForEachDataTypeHelper_(callback, int64_t, INT64);                          \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::complex<float>, COMPLEX64); \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::complex<double>, COMPLEX128);

#define _ForEachDataTypeNormal_(callback)                         \
  _ForEachDataTypeHelper_(callback, float, FP32);                 \
  _ForEachDataTypeHelper_(callback, double, FP64);                \
  _ForEachDataTypeHelper_(callback, int, INT32);                  \
  _ForEachDataTypeHelper_(callback, int64_t, INT64);              \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::float16, FP16); \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::bfloat16, BF16);

// For the use of thrust, as index-type elements can be only integers.
#define _ForEachDataTypeTiny_(callback)          \
  _ForEachDataTypeHelper_(callback, int, INT32); \
  _ForEachDataTypeHelper_(callback, int64_t, INT64);

// It's only for DataParallel in HIP, bf16 not support in HIP.
#define _ForEachDataTypeForHIP_(callback)                                     \
  _ForEachDataTypeHelper_(callback, float, FP32);                             \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::float16, FP16);             \
  _ForEachDataTypeHelper_(callback, double, FP64);                            \
  _ForEachDataTypeHelper_(callback, int, INT32);                              \
  _ForEachDataTypeHelper_(callback, int64_t, INT64);                          \
  _ForEachDataTypeHelper_(callback, bool, BOOL);                              \
  _ForEachDataTypeHelper_(callback, uint8_t, UINT8);                          \
  _ForEachDataTypeHelper_(callback, uint16_t, UINT16);                        \
  _ForEachDataTypeHelper_(callback, uint32_t, UINT32);                        \
  _ForEachDataTypeHelper_(callback, uint64_t, UINT64);                        \
  _ForEachDataTypeHelper_(callback, int16_t, INT16);                          \
  _ForEachDataTypeHelper_(callback, int8_t, INT8);                            \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::complex<float>, COMPLEX64); \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::complex<double>, COMPLEX128);

// complex and float8 are not supported on XPU.
#define _ForEachDataTypeForXPU_(callback)                          \
  _ForEachDataTypeHelper_(callback, float, FP32);                  \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::float16, FP16);  \
  _ForEachDataTypeHelper_(callback, ::phi::dtype::bfloat16, BF16); \
  _ForEachDataTypeHelper_(callback, double, FP64);                 \
  _ForEachDataTypeHelper_(callback, int, INT32);                   \
  _ForEachDataTypeHelper_(callback, int64_t, INT64);               \
  _ForEachDataTypeHelper_(callback, bool, BOOL);                   \
  _ForEachDataTypeHelper_(callback, uint8_t, UINT8);               \
  _ForEachDataTypeHelper_(callback, int16_t, INT16);               \
  _ForEachDataTypeHelper_(callback, int8_t, INT8);

#define DefineDataTypeTrait(cpp_type, proto_type)                         \
  template <>                                                             \
  struct DataTypeTrait<cpp_type> {                                        \
    constexpr static paddle::framework::proto::VarType::Type DataType() { \
      return proto_type;                                                  \
    }                                                                     \
  }

_ForEachDataType_(DefineDataTypeTrait);

#undef DefineDataTypeTrait

TEST_API extern VarType::Type ToDataType(std::type_index type);
extern std::type_index ToTypeIndex(VarType::Type type);

template <typename Visitor>
inline void VisitDataType(VarType::Type type, Visitor visitor) {
#define VisitDataTypeCallback(cpp_type, proto_type) \
  do {                                              \
    if (type == proto_type) {                       \
      visitor.template apply<cpp_type>();           \
      return;                                       \
    }                                               \
  } while (0)

  _ForEachDataType_(VisitDataTypeCallback);
#undef VisitDataTypeCallback
  PADDLE_THROW(common::errors::Unimplemented(
      "Not supported paddle::framework::proto::VarType::Type(%d) as data type.",
      static_cast<int>(type)));
}

template <typename Visitor>
inline void VisitDataTypeSmall(VarType::Type type, Visitor visitor) {
#define VisitDataTypeCallbackSmall(cpp_type, proto_type) \
  do {                                                   \
    if (type == proto_type) {                            \
      visitor.template apply<cpp_type>();                \
      return;                                            \
    }                                                    \
  } while (0)

  _ForEachDataTypeSmall_(VisitDataTypeCallbackSmall);
#undef VisitDataTypeCallbackSmall
}

// for normal dtype, int, int64, float, float64, float16
template <typename Visitor>
inline void VisitDataTypeNormal(VarType::Type type, Visitor visitor) {
#define VisitDataTypeCallbackNormal(cpp_type, proto_type) \
  do {                                                    \
    if (type == proto_type) {                             \
      visitor.template apply<cpp_type>();                 \
      return;                                             \
    }                                                     \
  } while (0)

  _ForEachDataTypeNormal_(VisitDataTypeCallbackNormal);
#undef VisitDataTypeCallbackNormal
}

template <typename Visitor>
inline void VisitIntDataType(VarType::Type type, Visitor visitor) {
#define VisitIntDataTypeCallback(cpp_type, proto_type) \
  do {                                                 \
    if (type == proto_type) {                          \
      visitor.template apply<cpp_type>();              \
      return;                                          \
    }                                                  \
  } while (0)

  _ForEachIntDataType_(VisitIntDataTypeCallback);

  PADDLE_THROW(common::errors::Unimplemented(
      "Expected integral data type, but got %s", VarDataTypeToString(type)));

#undef VisitIntDataTypeCallback
}

template <typename Visitor>
inline void VisitDataTypeTiny(VarType::Type type, Visitor visitor) {
#define VisitDataTypeCallbackTiny(cpp_type, proto_type) \
  do {                                                  \
    if (type == proto_type) {                           \
      visitor.template apply<cpp_type>();               \
      return;                                           \
    }                                                   \
  } while (0)

  _ForEachDataTypeTiny_(VisitDataTypeCallbackTiny);
#undef VisitDataTypeCallbackTiny
}

template <typename Visitor>
inline void VisitDataTypeForHIP(VarType::Type type, Visitor visitor) {
#define VisitDataTypeCallbackHIP(cpp_type, proto_type) \
  do {                                                 \
    if (type == proto_type) {                          \
      visitor.template apply<cpp_type>();              \
      return;                                          \
    }                                                  \
  } while (0)

  _ForEachDataTypeForHIP_(VisitDataTypeCallbackHIP);
#undef VisitDataTypeCallbackHIP
}

inline std::ostream& operator<<(std::ostream& out, const VarType::Type& type) {
  out << VarDataTypeToString(type);
  return out;
}

extern inline bool IsComplexType(const VarType::Type& type) {
  return (type == VarType::COMPLEX64 || type == VarType::COMPLEX128);
}

extern VarType::Type PromoteTypesIfComplexExists(const VarType::Type type_a,
                                                 const VarType::Type type_b);

extern inline VarType::Type ToComplexType(VarType::Type t) {
  switch (t) {
    case VarType::FP32:
      return VarType::COMPLEX64;
    case VarType::FP64:
      return VarType::COMPLEX128;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unknown real value data type (%s), now only support float32 and "
          "float64.",
          VarDataTypeToString(t)));
  }
}

extern inline VarType::Type ToRealType(VarType::Type t) {
  switch (t) {
    case VarType::COMPLEX64:
      return VarType::FP32;
    case VarType::COMPLEX128:
      return VarType::FP64;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unknown complex value data type (%s), now only support complex64 "
          "and complex128.",
          VarDataTypeToString(t)));
  }
}

}  // namespace phi
