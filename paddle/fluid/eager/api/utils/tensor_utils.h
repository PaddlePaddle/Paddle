// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/phi/api/all.h"

namespace egr {
namespace egr_utils_api {

// If and only if the tensor holds an AccumulationNode
// Then it's treated as a leaf tensor
bool IsLeafTensor(const paddle::experimental::Tensor& target);

paddle::experimental::Tensor CreateTensorWithValue(
    const phi::DDim& ddim, const paddle::platform::Place& place,
    const phi::DataType& dtype, const phi::DataLayout& layout, float value,
    bool is_leaf = true);

}  // namespace egr_utils_api
}  // namespace egr
