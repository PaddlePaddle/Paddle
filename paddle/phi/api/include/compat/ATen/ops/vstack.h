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
#include <ATen/ops/cat.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/unsqueeze.h>
#include <c10/util/Exception.h>
#include <vector>

namespace at {

inline at::Tensor vstack(at::TensorList tensors) {
  TORCH_CHECK(!tensors.empty(), "vstack expects a non-empty TensorList");

  std::vector<at::Tensor> processed;
  processed.reserve(tensors.size());

  for (const auto& t : tensors) {
    if (t.dim() == 0) {
      processed.push_back(t.reshape({1, 1}));
    } else if (t.dim() == 1) {
      processed.push_back(t.unsqueeze(0));
    } else {
      processed.push_back(t);
    }
  }

  return at::cat(processed, 0);
}

}  // namespace at
