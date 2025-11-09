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

#include <ATen/core/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <optional>

#include "paddle/phi/api/include/api.h"

namespace at {

// TODO(dev): The signature in PyTorch should be `at::Tensor cat(const
// at::ITensorListRef & tensors, int64_t dim=0)`
inline at::Tensor cat(const std::vector<at::Tensor>& tensors, int64_t dim = 0) {
  std::vector<paddle::Tensor> pd_tensors;
  pd_tensors.reserve(tensors.size());
  for (const auto& t : tensors) {
    pd_tensors.push_back(t._PD_GetInner());
  }
  return paddle::experimental::concat(pd_tensors, dim);
}

}  // namespace at

namespace torch {
using at::cat;
}  // namespace torch
