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

// chunk - splits tensor into chunks
inline std::vector<Tensor> chunk(const Tensor& self,
                                 int64_t chunks,
                                 int64_t dim = 0) {
  std::vector<Tensor> result;
  paddle::Tensor pd_tensor = self._PD_GetInner();
  int64_t dim_size = pd_tensor.dims().size() > 0 ? pd_tensor.dims()[dim] : 1;

  // PyTorch returns exactly 'chunks' number of tensors, even if some are empty
  // When chunks > dim_size, it returns dim_size non-empty tensors plus
  // (chunks - dim_size) empty tensors
  if (chunks > dim_size) {
    // First create non-empty chunks for existing elements
    for (int64_t i = 0; i < dim_size; ++i) {
      auto chunk_tensor =
          paddle::experimental::slice(pd_tensor, {dim}, {i}, {i + 1}, {1}, {});
      result.push_back(Tensor(chunk_tensor));
    }
    // Then add empty chunks
    for (int64_t i = dim_size; i < chunks; ++i) {
      // Create empty tensor with same shape except for the chunk dimension
      std::vector<int64_t> empty_shape;
      for (int64_t j = 0; j < pd_tensor.dims().size(); ++j) {
        if (j == dim) {
          empty_shape.push_back(0);
        } else {
          empty_shape.push_back(pd_tensor.dims()[j]);
        }
      }
      auto empty_tensor = paddle::experimental::empty(
          phi::IntArray(empty_shape), pd_tensor.dtype(), pd_tensor.place());
      result.push_back(Tensor(empty_tensor));
    }
    return result;
  }

  int64_t chunk_size = (dim_size + chunks - 1) / chunks;
  int64_t remaining = dim_size;

  for (int64_t i = 0; i < chunks && remaining > 0; ++i) {
    int64_t current_chunk_size = std::min(chunk_size, remaining);
    auto chunk_tensor =
        paddle::experimental::slice(pd_tensor,
                                    {dim},
                                    {i * chunk_size},
                                    {i * chunk_size + current_chunk_size},
                                    {1},
                                    {});
    result.push_back(Tensor(chunk_tensor));
    remaining -= current_chunk_size;
  }

  return result;
}

}  // namespace at

namespace at {

// Member function: Tensor::chunk
inline std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
  return at::chunk(*this, chunks, dim);
}

}  // namespace at
