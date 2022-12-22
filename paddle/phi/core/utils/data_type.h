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

// Here we can't depend on the fluid proto::VarType, so we use the dtype enum
// value directly. See also `assign_value_sig.cc`.
// proto::VarType::INT16 -> 1  -> phi::DataType::INT16
// proto::VarType::INT32 -> 2  -> phi::DataType::INT32
// proto::VarType::INT64 -> 3  -> phi::DataType::INT64
// proto::VarType::FP16 ->  4  -> phi::DataType::FLOAT16
// proto::VarType::FP32 ->  5  -> phi::DataType::FLOAT32
// proto::VarType::FP64 ->  6  -> phi::DataType::FLOAT64
// proto::VarType::UINT8 -> 20 -> phi::DataType::UINT8
static std::map<int, phi::DataType> var_type_map{{1, phi::DataType::INT16},
                                                 {2, phi::DataType::INT32},
                                                 {3, phi::DataType::INT64},
                                                 {4, phi::DataType::FLOAT16},
                                                 {5, phi::DataType::FLOAT32},
                                                 {6, phi::DataType::FLOAT64},
                                                 {20, phi::DataType::UINT8}};

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
}  // namespace phi
