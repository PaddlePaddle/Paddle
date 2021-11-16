/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "bfloat16.h"  // NOLINT
#include "complex.h"   // NOLINT
#include "float16.h"   // NOLINT

#include "paddle/pten/api/ext/exception.h"

namespace paddle {
namespace experimental {

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;
using float16 = ::paddle::platform::float16;
using bfloat16 = ::paddle::platform::bfloat16;

enum class DataType {
  UNDEFINED = 0,
  BOOL,
  INT8,   // Char
  UINT8,  // BYte
  INT16,
  INT32,
  UINT32,
  INT64,
  UINT64,
  BFLOAT16,
  FLOAT16,
  UINT16,
  FLOAT32,
  FLOAT64,
  COMPLEX64,
  COMPLEX128,
  NUM_DATA_TYPES
};

inline size_t SizeOf(DataType data_type) {
  switch (data_type) {
    case DataType::BOOL:
    case DataType::UINT8:
    case DataType::INT8:
      return 1;
    case DataType::BFLOAT16:
    case DataType::FLOAT16:
    case DataType::INT16:
    case DataType::UINT16:
      return 2;
    case DataType::FLOAT32:
    case DataType::INT32:
    case DataType::UINT32:
      return 4;
    case DataType::FLOAT64:
    case DataType::INT64:
    case DataType::UINT64:
    case DataType::COMPLEX64:
      return 8;
    case DataType::COMPLEX128:
      return 16;
    case DataType::UNDEFINED:
    case DataType::NUM_DATA_TYPES:
      PD_THROW("Data type `",
               static_cast<int>(data_type),
               "` is not supported by tensor.");
  }
  return 0;
}

#define PT_FOR_EACH_DATA_TYPE(_)    \
  _(bool, DataType::BOOL)           \
  _(int8_t, DataType::INT8)         \
  _(uint8_t, DataType::UINT8)       \
  _(int16_t, DataType::INT16)       \
  _(uint16_t, DataType::UINT16)     \
  _(int32_t, DataType::INT32)       \
  _(uint32_t, DataType::UINT32)     \
  _(int64_t, DataType::INT64)       \
  _(uint64_t, DataType::UINT64)     \
  _(bfloat16, DataType::BFLOAT16)   \
  _(float16, DataType::FLOAT16)     \
  _(float, DataType::FLOAT32)       \
  _(double, DataType::FLOAT64)      \
  _(complex64, DataType::COMPLEX64) \
  _(complex128, DataType::COMPLEX128)

template <DataType T>
struct DataTypeToCppType;

template <typename T>
struct CppTypeToDataType;

#define PT_SPECIALIZE_DataTypeToCppType(cpp_type, data_type) \
  template <>                                                \
  struct DataTypeToCppType<data_type> {                      \
    using type = cpp_type;                                   \
  };

PT_FOR_EACH_DATA_TYPE(PT_SPECIALIZE_DataTypeToCppType)

#undef PT_SPECIALIZE_DataTypeToCppType

#define PT_SPECIALIZE_CppTypeToDataType(cpp_type, data_type) \
  template <>                                                \
  struct CppTypeToDataType<cpp_type> {                       \
    constexpr static DataType Type() { return data_type; }   \
  };

PT_FOR_EACH_DATA_TYPE(PT_SPECIALIZE_CppTypeToDataType)

#undef PT_SPECIALIZE_CppTypeToDataType

inline std::ostream& operator<<(std::ostream& os, DataType dtype) {
  switch (dtype) {
    case DataType::UNDEFINED:
      os << "Undefined";
      break;
    case DataType::BOOL:
      os << "bool";
      break;
    case DataType::INT8:
      os << "int8";
      break;
    case DataType::UINT8:
      os << "uint8";
      break;
    case DataType::INT16:
      os << "int16";
      break;
    case DataType::UINT16:
      os << "uint16";
      break;
    case DataType::INT32:
      os << "int32";
      break;
    case DataType::UINT32:
      os << "uint32";
      break;
    case DataType::INT64:
      os << "int64";
      break;
    case DataType::UINT64:
      os << "uint64";
      break;
    case DataType::BFLOAT16:
      os << "bfloat16";
      break;
    case DataType::FLOAT16:
      os << "float16";
      break;
    case DataType::FLOAT32:
      os << "float32";
      break;
    case DataType::FLOAT64:
      os << "float64";
      break;
    case DataType::COMPLEX64:
      os << "complex64";
      break;
    case DataType::COMPLEX128:
      os << "complex128";
      break;
    default:
      PD_THROW("Invalid enum data type `", static_cast<int>(dtype), "`.");
  }
  return os;
}

}  // namespace experimental
}  // namespace paddle

namespace pten {
using DataType = paddle::experimental::DataType;

#define PTEN_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...) \
  case enum_type: {                                                         \
    using HINT = type;                                                      \
    __VA_ARGS__();                                                          \
    break;                                                                  \
  }

#define PTEN_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...) \
  PTEN_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, data_t, __VA_ARGS__)

#define PTEN_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                              \
  [&] {                                                                       \
    const auto& __dtype__ = TYPE;                                             \
    switch (__dtype__) {                                                      \
      PTEN_PRIVATE_CASE_TYPE(NAME, ::pten::DataType::BOOL, bool, __VA_ARGS__) \
      PTEN_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::pten::DataType::INT8, int8_t, __VA_ARGS__)                  \
      PTEN_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::pten::DataType::UINT8, uint8_t, __VA_ARGS__)                \
      PTEN_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::pten::DataType::INT16, int16_t, __VA_ARGS__)                \
      PTEN_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::pten::DataType::UINT16, uint16_t, __VA_ARGS__)              \
      PTEN_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::pten::DataType::INT32, int32_t, __VA_ARGS__)                \
      PTEN_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::pten::DataType::UINT32, uint32_t, __VA_ARGS__)              \
      PTEN_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::pten::DataType::INT64, int64_t, __VA_ARGS__)                \
      PTEN_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::pten::DataType::UINT64, uint64_t, __VA_ARGS__)              \
      PTEN_PRIVATE_CASE_TYPE(NAME,                                            \
                             ::pten::DataType::BFLOAT16,                      \
                             paddle::experimental::bfloat16,                  \
                             __VA_ARGS__)                                     \
      PTEN_PRIVATE_CASE_TYPE(NAME,                                            \
                             ::pten::DataType::FLOAT16,                       \
                             paddle::experimental::float16,                   \
                             __VA_ARGS__)                                     \
      PTEN_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::pten::DataType::FLOAT32, float, __VA_ARGS__)                \
      PTEN_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::pten::DataType::FLOAT64, double, __VA_ARGS__)               \
      PTEN_PRIVATE_CASE_TYPE(NAME,                                            \
                             ::pten::DataType::COMPLEX64,                     \
                             paddle::experimental::complex64,                 \
                             __VA_ARGS__)                                     \
      PTEN_PRIVATE_CASE_TYPE(NAME,                                            \
                             ::pten::DataType::COMPLEX128,                    \
                             paddle::experimental::complex128,                \
                             __VA_ARGS__)                                     \
      default:                                                                \
        PADDLE_THROW(paddle::platform::errors::InvalidArgument(               \
            "Invalid enum data type `%d`.", static_cast<int>(__dtype__)));    \
    }                                                                         \
  }()
}  // namespace pten

namespace paddle {
// In order to be compatible with the original custom operator Tensor interface
using DataType = paddle::experimental::DataType;
using bfloat16 = paddle::experimental::bfloat16;
using complex64 = paddle::experimental::complex64;
using complex128 = paddle::experimental::complex128;
using float16 = paddle::experimental::float16;
}  // namespace paddle
