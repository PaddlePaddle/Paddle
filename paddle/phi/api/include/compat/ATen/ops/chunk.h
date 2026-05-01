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
  if (chunks <= 0) {
    PD_THROW("chunk expects chunks to be greater than 0, got ", chunks);
  }

  std::vector<Tensor> result;
  paddle::Tensor pd_tensor = self._PD_GetInner();

  int64_t rank = static_cast<int64_t>(pd_tensor.dims().size());
  if (rank == 0) {
    PD_THROW("chunk expects at least a 1-dimensional tensor");
  }

  int64_t original_dim = dim;
  if (dim < 0) {
    dim += rank;
  }
  if (dim < 0 || dim >= rank) {
    PD_THROW("Dimension out of range (expected to be in range of [",
             -rank,
             ", ",
             rank - 1,
             "], but got ",
             original_dim,
             ")");
  }

  int64_t dim_size = pd_tensor.dims()[dim];

  if (dim_size == 0) {
    for (int64_t i = 0; i < chunks; ++i) {
      auto chunk_tensor =
          paddle::experimental::slice(pd_tensor, {dim}, {0}, {0}, {1}, {});
      result.push_back(Tensor(chunk_tensor));
    }
    return result;
  }

  // PyTorch returns at most 'dim_size' non-empty chunks when chunks > dim_size
  if (chunks > dim_size) {
    chunks = dim_size;
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
