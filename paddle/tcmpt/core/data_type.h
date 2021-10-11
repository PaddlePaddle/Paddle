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

#include "paddle/tcmpt/common/data_layout.h"
#include "paddle/tcmpt/common/data_type.h"

namespace paddle {
namespace tcmpt {

using DataType = experimental::DataType;
using DataLayout = experimental::DataLayout;

size_t SizeOf(DataType type);

template <typename T>
struct DataTypeTrait {};

#define DefineDataTypeTrait(cpp_type, proto_type)                \
  template <>                                                    \
  struct DataTypeTrait<cpp_type> {                               \
    constexpr static DataType data_type() { return proto_type; } \
  };

DefineDataTypeTrait(void, DataType::INVALID);
DefineDataTypeTrait(int8_t, DataType::INT8);
DefineDataTypeTrait(uint8_t, DataType::UINT8);
DefineDataTypeTrait(int16_t, DataType::INT16);
DefineDataTypeTrait(uint16_t, DataType::UINT16);
DefineDataTypeTrait(int32_t, DataType::INT32);
DefineDataTypeTrait(uint32_t, DataType::UINT32);
DefineDataTypeTrait(int64_t, DataType::INT64);
DefineDataTypeTrait(uint64_t, DataType::UINT64);
DefineDataTypeTrait(float, DataType::FLOAT32);
DefineDataTypeTrait(double, DataType::FLOAT64);

#undef DefineDataTypeTrait

}  // namespace tcmpt
}  // namespace paddle
