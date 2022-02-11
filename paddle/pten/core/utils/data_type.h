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
#include <string>
#include <typeindex>

#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/enforce.h"
#include "paddle/pten/kernels/funcs/eigen/extensions.h"

namespace pten {

#define _PtenForEachDataTypeHelper_(callback, cpp_type, data_type) \
  callback(cpp_type, data_type);

#define _PtenForEachDataType_(callback)                                   \
  _PtenForEachDataTypeHelper_(callback, float, DataType::FLOAT32);        \
  _PtenForEachDataTypeHelper_(                                            \
      callback, ::paddle::platform::float16, DataType::FLOAT16);          \
  _PtenForEachDataTypeHelper_(                                            \
      callback, ::paddle::platform::bfloat16, DataType::BFLOAT16);        \
  _PtenForEachDataTypeHelper_(callback, double, DataType::FLOAT64);       \
  _PtenForEachDataTypeHelper_(callback, int, DataType::INT32);            \
  _PtenForEachDataTypeHelper_(callback, int64_t, DataType::INT64);        \
  _PtenForEachDataTypeHelper_(callback, bool, DataType::BOOL);            \
  _PtenForEachDataTypeHelper_(callback, uint8_t, DataType::UINT8);        \
  _PtenForEachDataTypeHelper_(callback, int16_t, DataType::INT16);        \
  _PtenForEachDataTypeHelper_(callback, int8_t, DataType::INT8);          \
  _PtenForEachDataTypeHelper_(                                            \
      callback, ::paddle::platform::complex<float>, DataType::COMPLEX64); \
  _PtenForEachDataTypeHelper_(                                            \
      callback, ::paddle::platform::complex<double>, DataType::COMPLEX128);

template <typename Visitor>
inline void VisitDataType(pten::DataType type, Visitor visitor) {
#define PtenVisitDataTypeCallback(cpp_type, data_type) \
  do {                                                 \
    if (type == data_type) {                           \
      visitor.template apply<cpp_type>();              \
      return;                                          \
    }                                                  \
  } while (0)

  _PtenForEachDataType_(PtenVisitDataTypeCallback);
#undef PtenVisitDataTypeCallback
  PADDLE_THROW(pten::errors::Unimplemented(
      "Not supported proto::VarType::Type(%d) as data type.",
      static_cast<int>(type)));
}
}  // namespace pten
