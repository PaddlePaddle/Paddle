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

#include "paddle/phi/api/include/api.h"

namespace at {

// expand - expands tensor to new size
// If dimensions are not compatible for expand (i.e., non-1 dims don't match),
// falls back to tile operation to replicate the tensor, then slices to exact
// size
inline Tensor expand(const Tensor& self,
                     at::IntArrayRef size,
                     bool implicit = false) {
  paddle::Tensor pd_tensor = self._PD_GetInner();

  std::vector<int64_t> current_size_vec;
  for (int64_t i = 0; i < pd_tensor.dims().size(); ++i) {
    current_size_vec.push_back(pd_tensor.dims()[i]);
  }

  std::vector<int64_t> target_size_vec;
  for (int64_t i = 0; i < size.size(); ++i) {
    target_size_vec.push_back(size[i]);
  }

  // Calculate repeat factors
  int64_t ndims = target_size_vec.size();
  int64_t current_ndims = current_size_vec.size();
  int64_t start_dim = ndims - current_ndims;

  std::vector<int64_t> repeat_vec(ndims, 1);
  bool need_tile = false;
  for (int64_t i = 0; i < current_ndims; ++i) {
    int64_t target_dim = start_dim + i;
    if (target_dim >= 0 && target_dim < ndims) {
      if (target_size_vec[target_dim] == current_size_vec[i]) {
        repeat_vec[target_dim] = 1;
      } else if (current_size_vec[i] == 1) {
        repeat_vec[target_dim] = target_size_vec[target_dim];
      } else {
        // Cannot expand directly - need to use tile
        need_tile = true;
        // Calculate how many times to repeat to cover the target size
        repeat_vec[target_dim] =
            (target_size_vec[target_dim] + current_size_vec[i] - 1) /
            current_size_vec[i];
      }
    }
  }

  paddle::Tensor result;
  if (need_tile) {
    // Use tile to get at least the target size
    result = paddle::experimental::tile(pd_tensor, phi::IntArray(repeat_vec));

    // If tiled result is larger than target, slice to exact size
    std::vector<int64_t> tiled_size;
    for (int64_t i = 0; i < result.dims().size(); ++i) {
      tiled_size.push_back(result.dims()[i]);
    }

    bool need_slice = false;
    for (int64_t i = 0; i < ndims; ++i) {
      if (tiled_size[i] > target_size_vec[i]) {
        need_slice = true;
        break;
      }
    }

    if (need_slice) {
      std::vector<int64_t> starts_vec(ndims, 0);
      std::vector<int64_t> ends_vec = target_size_vec;
      std::vector<int64_t> axes_vec;
      for (int64_t i = 0; i < ndims; ++i) {
        axes_vec.push_back(i);
      }
      result = paddle::experimental::slice(result,
                                           axes_vec,
                                           phi::IntArray(starts_vec),
                                           phi::IntArray(ends_vec),
                                           {1},
                                           {});
    }
  } else {
    result = paddle::experimental::tile(pd_tensor, phi::IntArray(repeat_vec));
  }

  return Tensor(result);
}

}  // namespace at
