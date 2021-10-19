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

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace experimental {

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;
using float16 = ::paddle::platform::float16;
using bfloat16 = ::paddle::platform::bfloat16;

enum class DataType {
  kUndef = 0,
  kBOOL,
  kINT8,   // Char
  kUINT8,  // BYte
  kINT16,
  kINT32,
  kUINT32,
  kINT64,
  kUINT64,
  kBFLOAT16,
  kFLOAT16,
  kUINT16,
  kFLOAT32,
  kFLOAT64,
  kCOMPLEX64,
  kCOMPLEX128,
  kNumDataTypes
};

inline size_t SizeOf(DataType data_type) {
  switch (data_type) {
    case DataType::kBOOL:
    case DataType::kUINT8:
    case DataType::kINT8:
      return 1;
    case DataType::kFLOAT16:
    case DataType::kINT16:
    case DataType::kUINT16:
      return 2;
    case DataType::kFLOAT32:
    case DataType::kINT32:
    case DataType::kUINT32:
      return 4;
    case DataType::kFLOAT64:
    case DataType::kINT64:
    case DataType::kUINT64:
      return 8;
    case DataType::kUndef:
    case DataType::kBFLOAT16:
    case DataType::kCOMPLEX64:
    case DataType::kCOMPLEX128:
    case DataType::kNumDataTypes:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type %d is not supported by tensor.",
          static_cast<int>(data_type)));
      return 0;
  }
}

#define PT_FOR_EACH_DATA_TYPE(_)     \
  _(bool, DataType::kBOOL)           \
  _(int8_t, DataType::kINT8)         \
  _(uint8_t, DataType::kUINT8)       \
  _(int16_t, DataType::kINT16)       \
  _(int, DataType::kINT32)           \
  _(int64_t, DataType::kINT64)       \
  _(bfloat16, DataType::kBFLOAT16)   \
  _(float16, DataType::kFLOAT16)     \
  _(float, DataType::kFLOAT32)       \
  _(double, DataType::kFLOAT64)      \
  _(complex64, DataType::kCOMPLEX64) \
  _(complex128, DataType::kCOMPLEX128)

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
    case DataType::kUndef:
      os << "Undefined";
      break;
    case DataType::kBOOL:
      os << "bool";
      break;
    case DataType::kINT8:
      os << "int8";
      break;
    case DataType::kUINT8:
      os << "uint8";
      break;
    case DataType::kINT16:
      os << "int16";
      break;
    case DataType::kINT32:
      os << "int32";
      break;
    case DataType::kINT64:
      os << "int64";
      break;
    case DataType::kBFLOAT16:
      os << "bfloat16";
      break;
    case DataType::kFLOAT16:
      os << "float16";
      break;
    case DataType::kFLOAT32:
      os << "float32";
      break;
    case DataType::kFLOAT64:
      os << "float64";
      break;
    case DataType::kCOMPLEX64:
      os << "complex64";
      break;
    case DataType::kCOMPLEX128:
      os << "complex128";
      break;
    default:
      // TODO(chenweihang): change to enforce later
      throw std::runtime_error("Invalid DataType type.");
  }
  return os;
}

inline DataType& operator++(DataType& dtype, int) {
  dtype =
      DataType(static_cast<std::underlying_type<DataType>::type>(dtype) + 1);
  return dtype;
}

}  // namespace experimental
}  // namespace paddle

namespace pt {
using DataType = paddle::experimental::DataType;
}
