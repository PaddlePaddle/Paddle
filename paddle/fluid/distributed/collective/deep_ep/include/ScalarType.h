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

namespace deep_ep::detail {

using ScalarType = phi::DataType;

constexpr auto kInt32 = phi::DataType::INT32;
constexpr auto kBool = phi::DataType::BOOL;
constexpr auto kFloat32 = phi::DataType::FLOAT32;
constexpr auto kByte = phi::DataType::INT8;
}  // namespace deep_ep::detail
