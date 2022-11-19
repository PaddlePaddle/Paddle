/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include <iostream>
#include <map>
#include <string>
#include <typeindex>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

namespace phi {

#define _PhiForEachDataTypeHelper_(callback, cpp_type, data_type) \
  callback(cpp_type, data_type);

#define _PhiForEachDataType_(callback)                              \
  _PhiForEachDataTypeHelper_(callback, float, DataType::FLOAT32);   \
  _PhiForEachDataTypeHelper_(                                       \
      callback, ::phi::dtype::float16, DataType::FLOAT16);          \
  _PhiForEachDataTypeHelper_(                                       \
      callback, ::phi::dtype::bfloat16, DataType::BFLOAT16);        \
  _PhiForEachDataTypeHelper_(callback, double, DataType::FLOAT64);  \
  _PhiForEachDataTypeHelper_(callback, int, DataType::INT32);       \
  _PhiForEachDataTypeHelper_(callback, int64_t, DataType::INT64);   \
  _PhiForEachDataTypeHelper_(callback, bool, DataType::BOOL);       \
  _PhiForEachDataTypeHelper_(callback, uint8_t, DataType::UINT8);   \
  _PhiForEachDataTypeHelper_(callback, int16_t, DataType::INT16);   \
  _PhiForEachDataTypeHelper_(callback, int8_t, DataType::INT8);     \
  _PhiForEachDataTypeHelper_(                                       \
      callback, ::phi::dtype::complex<float>, DataType::COMPLEX64); \
  _PhiForEachDataTypeHelper_(                                       \
      callback, ::phi::dtype::complex<double>, DataType::COMPLEX128);

#define _PhiForEachDataTypeTiny_(callback)                    \
  _PhiForEachDataTypeHelper_(callback, int, DataType::INT32); \
  _PhiForEachDataTypeHelper_(callback, int64_t, DataType::INT64);

template <typename Visitor>
inline void VisitDataType(phi::DataType type, Visitor visitor) {
#define PhiVisitDataTypeCallback(cpp_type, data_type) \
  do {                                                \
    if (type == data_type) {                          \
      visitor.template apply<cpp_type>();             \
      return;                                         \
    }                                                 \
  } while (0)

  _PhiForEachDataType_(PhiVisitDataTypeCallback);
#undef PhiVisitDataTypeCallback
  PADDLE_THROW(phi::errors::Unimplemented(
      "Not supported phi::DataType(%d) as data type.", static_cast<int>(type)));
}

template <typename Visitor>
inline void VisitDataTypeTiny(phi::DataType type, Visitor visitor) {
#define PhiVisitDataTypeCallbackTiny(cpp_type, data_type) \
  do {                                                    \
    if (type == data_type) {                              \
      visitor.template apply<cpp_type>();                 \
      return;                                             \
    }                                                     \
  } while (0)

  _PhiForEachDataTypeTiny_(PhiVisitDataTypeCallbackTiny);
#undef PhiVisitDataTypeCallbackTiny
  PADDLE_THROW(phi::errors::Unimplemented(
      "Not supported phi::DataType(%d) as data type.", static_cast<int>(type)));
}

inline bool IsComplexType(const DataType& type) {
  return (type == DataType::COMPLEX64 || type == DataType::COMPLEX128);
}

inline DataType ToComplexType(const DataType& type) {
  switch (type) {
    case DataType::FLOAT32:
      return DataType::COMPLEX64;
    case DataType::FLOAT64:
      return DataType::COMPLEX128;
    default:
      PADDLE_THROW(errors::Unimplemented(
          "Can not transform data type (%s) to complex type, now only support "
          "float32 and float64 real value.",
          type));
  }
}

inline DataType ToRealType(const DataType& type) {
  switch (type) {
    case DataType::COMPLEX64:
      return DataType::FLOAT32;
    case DataType::COMPLEX128:
      return DataType::FLOAT64;
    default:
      PADDLE_THROW(errors::Unimplemented(
          "Can not transform data type (%s) to real type, now only support "
          "complex64 and complex128 value.",
          type));
  }
}

// We can't depend on the fluid proto::VarType
// so we copy part of fluid proto::VarType here.
enum ProtoVarType {
  BOOL = 0,
  INT16 = 1,
  INT32 = 2,
  INT64 = 3,
  FP16 = 4,
  FP32 = 5,
  FP64 = 6,
  UINT8 = 20,
  INT8 = 21,
  BF16 = 22,
  COMPLEX64 = 23,
  COMPLEX128 = 24,
  PSTRING = 29
};

inline DataType TransToPhiDataType(const int& dtype) {
  // Set the order of case branches according to the frequency with
  // the data type is used
  switch (dtype) {
    case ProtoVarType::FP32:
      return DataType::FLOAT32;
    case ProtoVarType::FP64:
      return DataType::FLOAT64;
    case ProtoVarType::INT64:
      return DataType::INT64;
    case ProtoVarType::INT32:
      return DataType::INT32;
    case ProtoVarType::INT8:
      return DataType::INT8;
    case ProtoVarType::UINT8:
      return DataType::UINT8;
    case ProtoVarType::INT16:
      return DataType::INT16;
    case ProtoVarType::COMPLEX64:
      return DataType::COMPLEX64;
    case ProtoVarType::COMPLEX128:
      return DataType::COMPLEX128;
    case ProtoVarType::FP16:
      return DataType::FLOAT16;
    case ProtoVarType::BF16:
      return DataType::BFLOAT16;
    case ProtoVarType::BOOL:
      return DataType::BOOL;
    case ProtoVarType::PSTRING:
      return DataType::PSTRING;
    default:
      return DataType::UNDEFINED;
  }
}

inline int TransToProtoVarType(const DataType& dtype) {
  // Set the order of case branches according to the frequency with
  // the data type is used
  switch (dtype) {
    case DataType::FLOAT32:
      return ProtoVarType::FP32;
    case DataType::FLOAT64:
      return ProtoVarType::FP64;
    case DataType::INT64:
      return ProtoVarType::INT64;
    case DataType::INT32:
      return ProtoVarType::INT32;
    case DataType::INT8:
      return ProtoVarType::INT8;
    case DataType::UINT8:
      return ProtoVarType::UINT8;
    case DataType::INT16:
      return ProtoVarType::INT16;
    case DataType::COMPLEX64:
      return ProtoVarType::COMPLEX64;
    case DataType::COMPLEX128:
      return ProtoVarType::COMPLEX128;
    case DataType::FLOAT16:
      return ProtoVarType::FP16;
    case DataType::BFLOAT16:
      return ProtoVarType::BF16;
    case DataType::BOOL:
      return ProtoVarType::BOOL;
    case DataType::PSTRING:
      return ProtoVarType::PSTRING;
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported data type `%s` when casting it into "
          "paddle data type.",
          dtype));
  }
}

}  // namespace phi
