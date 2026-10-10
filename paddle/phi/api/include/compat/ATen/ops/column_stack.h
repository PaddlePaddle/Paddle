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
#include <vector>

#include "paddle/phi/api/include/api.h"

namespace at {

inline at::Tensor column_stack(at::TensorList tensors) {
  PD_CHECK(!tensors.empty(), "column_stack expects a non-empty TensorList");

  std::vector<paddle::Tensor> pd_tensors;
  pd_tensors.reserve(tensors.size());
  for (const auto& t : tensors) {
    if (t.dim() <= 1) {
      pd_tensors.push_back(paddle::experimental::reshape(
          t._PD_GetInner(), phi::IntArray(std::vector<int64_t>{t.numel(), 1})));
    } else {
      pd_tensors.push_back(t._PD_GetInner());
    }
  }
  return paddle::experimental::concat(pd_tensors, 1);
}

}  // namespace at
