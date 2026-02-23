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
  for (int64_t i = 0; i < current_ndims; ++i) {
    int64_t target_dim = start_dim + i;
    if (target_dim >= 0 && target_dim < ndims) {
      if (target_size_vec[target_dim] == current_size_vec[i]) {
        repeat_vec[target_dim] = 1;
      } else if (current_size_vec[i] == 1) {
        repeat_vec[target_dim] = target_size_vec[target_dim];
      } else {
        PD_THROW("expand size mismatch");
      }
    }
  }

  auto result =
      paddle::experimental::tile(pd_tensor, phi::IntArray(repeat_vec));
  return Tensor(result);
}

}  // namespace at
