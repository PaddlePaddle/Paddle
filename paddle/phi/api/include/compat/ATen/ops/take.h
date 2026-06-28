// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// take - Returns a new tensor with the elements of input at the given indices.
// The input tensor is treated as if it were viewed as a 1-D tensor.
// The result takes the same shape as the indices.
//
// PyTorch behavior:
// - index must be a Long tensor
// - output dtype matches input dtype
// - output shape matches index shape
inline at::Tensor take(const at::Tensor& self, const at::Tensor& index) {
  // PyTorch requires index to be a Long (INT64) tensor
  TORCH_CHECK(index.scalar_type() == at::ScalarType::Long,
              "take(): Expected a long tensor for index, but got ",
              index.scalar_type());

  TORCH_CHECK(self.numel() > 0 || index.numel() == 0,
              "take(): cannot take from an empty tensor with non-empty index");

  // Flatten self to 1D (treat as 1-D tensor for indexing).
  auto flattened =
      paddle::experimental::reshape(self._PD_GetInner(), phi::IntArray({-1}));

  // Record the original index shape for reshaping the result
  auto index_sizes = index.sizes();
  std::vector<int64_t> index_shape_vec(index_sizes.begin(), index_sizes.end());

  // Flatten index to 1D. take_along_axis performs unified CPU/GPU bounds
  // checking and normalizes legal negative indices on device.
  auto flattened_index =
      paddle::experimental::reshape(index._PD_GetInner(), phi::IntArray({-1}));

  // Use take_along_axis along axis 0 to pick elements by index.
  auto selected =
      paddle::experimental::take_along_axis(flattened, flattened_index, 0);

  // Reshape result to match original index shape
  if (index_shape_vec.empty()) {
    // Scalar index: return scalar (0-D tensor)
    return Tensor(paddle::experimental::reshape(
        selected, phi::IntArray(std::vector<int64_t>{})));
  }
  return Tensor(
      paddle::experimental::reshape(selected, phi::IntArray(index_shape_vec)));
}

inline at::Tensor Tensor::take(const at::Tensor& index) const {
  return at::take(*this, index);
}

}  // namespace at
