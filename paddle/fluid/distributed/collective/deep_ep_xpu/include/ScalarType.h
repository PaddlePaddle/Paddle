// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/data_type.h"

namespace paddle::deep_ep::detail {

using ScalarType = phi::DataType;

constexpr auto kInt32 = DataType::INT32;
constexpr auto kInt64 = DataType::INT64;
constexpr auto kBool = DataType::BOOL;
// constexpr auto kFloat8_e4m3fn = DataType::FLOAT8_E4M3FN;
constexpr auto kBFloat16 = DataType::BFLOAT16;
constexpr auto kFloat32 = DataType::FLOAT32;
constexpr auto kByte = DataType::INT8;

}  // namespace paddle::deep_ep::detail
