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

#define _ForEachDataTypeHelper_(callback, cpp_type, proto_type) \
  callback(cpp_type, proto_type);

#define _ForEachDataType_(callback)                                      \
  _ForEachDataTypeHelper_(callback, float, FP32);                        \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::float16, FP16);  \
  _ForEachDataTypeHelper_(callback, ::paddle::platform::bfloat16, BF16); \
  _ForEachDataTypeHelper_(callback, double, FP64);                       \
  _ForEachDataTypeHelper_(callback, int, INT32);                         \
  _ForEachDataTypeHelper_(callback, int64_t, INT64);                     \
  _ForEachDataTypeHelper_(callback, bool, BOOL);                         \
  _ForEachDataTypeHelper_(callback, uint8_t, UINT8);                     \
  _ForEachDataTypeHelper_(callback, int16_t, INT16);                     \
  _ForEachDataTypeHelper_(callback, int8_t, INT8);                       \
  _ForEachDataTypeHelper_(                                               \
      callback, ::paddle::platform::complex<float>, COMPLEX64);          \
  _ForEachDataTypeHelper_(                                               \
      callback, ::paddle::platform::complex<double>, COMPLEX128);

template <typename Visitor>
inline void VisitDataType(pten::DataType type, Visitor visitor) {
#define VisitDataTypeCallback(cpp_type, proto_type) \
  do {                                              \
    if (type == proto_type) {                       \
      visitor.template apply<cpp_type>();           \
      return;                                       \
    }                                               \
  } while (0)

  _ForEachDataType_(VisitDataTypeCallback);
#undef VisitDataTypeCallback
  PADDLE_THROW(platform::errors::Unimplemented(
      "Not supported proto::VarType::Type(%d) as data type.",
      static_cast<int>(type)));
}
}  // namespace pten
