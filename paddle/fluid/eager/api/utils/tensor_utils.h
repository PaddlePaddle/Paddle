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
#include "paddle/pten/api/all.h"

namespace egr {

// If and only if the tensor holds an AccumulationNode
// Then it's treated as a leaf tensor
bool IsLeafTensor(const egr::EagerTensor& target);

egr::EagerTensor CreateTensorWithValue(const pten::DDim& ddim,
                                       const paddle::platform::Place& place,
                                       const pten::DataType& dtype,
                                       const pten::DataLayout& layout,
                                       float value, bool is_leaf = true);

}  // namespace egr
