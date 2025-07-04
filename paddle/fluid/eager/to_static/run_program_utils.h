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

#include <cstdint>
#include <string>
#include <vector>
#include "paddle/phi/api/include/tensor.h"

namespace egr::to_static {
bool IsFakeValueName(const std::string& name);

int64_t hash_with_seed(int64_t value, int64_t seed);

bool IsVariableRefArray(const paddle::Tensor& tensor);

std::vector<paddle::Tensor> DereferenceTensors(
    const std::vector<paddle::Tensor*>& tensor_ptr);
}  // namespace egr::to_static
