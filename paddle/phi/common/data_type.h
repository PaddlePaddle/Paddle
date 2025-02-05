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

#include "paddle/common/exception.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/float8_e4m3fn.h"
#include "paddle/phi/common/float8_e5m2.h"
#include "paddle/utils/test_macros.h"

namespace phi {
namespace dtype {
class pstring;
}  // namespace dtype
}  // namespace phi

namespace phi {

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;
using float16 = ::phi::dtype::float16;
using bfloat16 = ::phi::dtype::bfloat16;
using float8_e4m3fn = ::phi::dtype::float8_e4m3fn;
using float8_e5m2 = ::phi::dtype::float8_e5m2;
using pstring = ::phi::dtype::pstring;

// The enum value are consistent with jit/property.proto
enum class TEST_API DataType {
  UNDEFINED = 0,

  BOOL,

  UINT8,  // Byte
  INT8,   // Char
  UINT16,
  INT16,
  UINT32,
  INT32,
  UINT64,
  INT64,

  FLOAT32,
  FLOAT64,

  COMPLEX64,
  COMPLEX128,

  // In Paddle 2.3, we add a new type of Tensor, StringTensor, which is designed
  // for string data management. We design the dtype of StringTensor, pstring.
  // In order to express a unique data dtype of StringTensor, we add
  // DataType::PSTRING.
  PSTRING,

  // IEEE754 half-precision floating-point format (16 bits wide).
  // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  FLOAT16,

  // Non-IEEE floating-point format based on IEEE754 single-precision
  // floating-point number truncated to 16 bits.
  // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
  BFLOAT16,

  // This format has 1 sign bit, 4 exponent bits, and 3 mantissa bits.
  FLOAT8_E4M3FN,
  // This format has 1 sign bit, 5 exponent bits, and 2 mantissa bits.
  FLOAT8_E5M2,

  NUM_DATA_TYPES,
  // See Note [ Why we need ALL in basic kernel key member? ]
  ALL_DTYPE = UNDEFINED,
};

inline size_t SizeOf(DataType data_type) {
  switch (data_type) {
    case DataType::BOOL:
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::FLOAT8_E4M3FN:
    case DataType::FLOAT8_E5M2:
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
    case DataType::PSTRING:
      return 48;
    case DataType::UNDEFINED:
      return 0;
    case DataType::NUM_DATA_TYPES:
      PD_THROW("Data type `",
               static_cast<int>(data_type),
               "` is not supported by tensor.");
  }
  return 0;
}

#define PD_FOR_EACH_DATA_TYPE(_)            \
  _(bool, DataType::BOOL)                   \
  _(int8_t, DataType::INT8)                 \
  _(uint8_t, DataType::UINT8)               \
  _(int16_t, DataType::INT16)               \
  _(uint16_t, DataType::UINT16)             \
  _(int32_t, DataType::INT32)               \
  _(uint32_t, DataType::UINT32)             \
  _(int64_t, DataType::INT64)               \
  _(uint64_t, DataType::UINT64)             \
  _(bfloat16, DataType::BFLOAT16)           \
  _(float8_e4m3fn, DataType::FLOAT8_E4M3FN) \
  _(float8_e5m2, DataType::FLOAT8_E5M2)     \
  _(float16, DataType::FLOAT16)             \
  _(float, DataType::FLOAT32)               \
  _(double, DataType::FLOAT64)              \
  _(complex64, DataType::COMPLEX64)         \
  _(complex128, DataType::COMPLEX128)       \
  _(pstring, DataType::PSTRING)

template <DataType T>
struct DataTypeToCppType;

template <typename T>
struct CppTypeToDataType;

#define PD_SPECIALIZE_DataTypeToCppType(cpp_type, data_type) \
  template <>                                                \
  struct DataTypeToCppType<data_type> {                      \
    using type = cpp_type;                                   \
  };

PD_FOR_EACH_DATA_TYPE(PD_SPECIALIZE_DataTypeToCppType)

#undef PD_SPECIALIZE_DataTypeToCppType

#define PD_SPECIALIZE_CppTypeToDataType(cpp_type, data_type) \
  template <>                                                \
  struct CppTypeToDataType<cpp_type> {                       \
    constexpr static DataType Type() { return data_type; }   \
  };

PD_FOR_EACH_DATA_TYPE(PD_SPECIALIZE_CppTypeToDataType)

#undef PD_SPECIALIZE_CppTypeToDataType

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
    case DataType::FLOAT8_E4M3FN:
      os << "float8_e4m3fn";
      break;
    case DataType::FLOAT8_E5M2:
      os << "float8_e5m2";
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
    case DataType::PSTRING:
      os << "pstring";
      break;
    default:
      PD_THROW("Invalid enum data type `", static_cast<int>(dtype), "`.");
  }
  return os;
}

inline std::string DataTypeToString(const DataType& dtype) {
  switch (dtype) {
    case DataType::UNDEFINED:
      return "Undefined(ALL_DTYPE)";
    case DataType::BOOL:
      return "bool";
    case DataType::INT8:
      return "int8";
    case DataType::UINT8:
      return "uint8";
    case DataType::INT16:
      return "int16";
    case DataType::UINT16:
      return "uint16";
    case DataType::INT32:
      return "int32";
    case DataType::UINT32:
      return "uint32";
    case DataType::INT64:
      return "int64";
    case DataType::UINT64:
      return "uint64";
    case DataType::FLOAT8_E4M3FN:
      return "float8_e4m3fn";
    case DataType::FLOAT8_E5M2:
      return "float8_e5m2";
    case DataType::BFLOAT16:
      return "bfloat16";
    case DataType::FLOAT16:
      return "float16";
    case DataType::FLOAT32:
      return "float32";
    case DataType::FLOAT64:
      return "float64";
    case DataType::COMPLEX64:
      return "complex64";
    case DataType::COMPLEX128:
      return "complex128";
    case DataType::PSTRING:
      return "pstring";
    default:
      PD_THROW("Invalid enum data type `", static_cast<int>(dtype), "`.");
  }
}

inline DataType StringToDataType(const std::string& dtype) {
  if (dtype == "Undefined(ALL_DTYPE)") {
    return DataType::UNDEFINED;
  } else if (dtype == "bool") {
    return DataType::BOOL;
  } else if (dtype == "int8") {
    return DataType::INT8;
  } else if (dtype == "uint8") {
    return DataType::UINT8;
  } else if (dtype == "int16") {
    return DataType::INT16;
  } else if (dtype == "uint16") {
    return DataType::UINT16;
  } else if (dtype == "int32") {
    return DataType::INT32;
  } else if (dtype == "uint32") {
    return DataType::UINT32;
  } else if (dtype == "int64") {
    return DataType::INT64;
  } else if (dtype == "uint64") {
    return DataType::UINT64;
  } else if (dtype == "bfloat16") {
    return DataType::BFLOAT16;
  } else if (dtype == "float16") {
    return DataType::FLOAT16;
  } else if (dtype == "float32") {
    return DataType::FLOAT32;
  } else if (dtype == "float64") {
    return DataType::FLOAT64;
  } else if (dtype == "complex64") {
    return DataType::COMPLEX64;
  } else if (dtype == "complex128") {
    return DataType::COMPLEX128;
  } else if (dtype == "pstring") {
    return DataType::PSTRING;
  } else {
    PD_THROW("Invalid enum data type `", dtype, "`.");
  }
}

}  // namespace phi

namespace paddle {
// In order to be compatible with the original custom operator Tensor interface
using DataType = phi::DataType;
using float8_e4m3fn = phi::float8_e4m3fn;
using float8_e5m2 = phi::float8_e5m2;
using bfloat16 = phi::bfloat16;
using complex64 = phi::complex64;
using complex128 = phi::complex128;
using float16 = phi::float16;
using pstring = phi::pstring;

}  // namespace paddle
