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
}  // namespace phi
