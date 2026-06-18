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

namespace at {

inline at::Tensor unflatten(const at::Tensor& self,
                            int64_t dim,
                            at::IntArrayRef sizes) {
  // Compute the new shape by replacing the dimension at 'dim' with 'sizes'
  int64_t ndim = self._PD_GetInner().dims().size();
  int64_t actual_dim = dim < 0 ? dim + ndim : dim;
  std::vector<int64_t> new_shape;
  for (int64_t i = 0; i < ndim; ++i) {
    if (i == actual_dim) {
      for (auto s : sizes) {
        new_shape.push_back(s);
      }
    } else {
      new_shape.push_back(self._PD_GetInner().dims()[i]);
    }
  }
  return Tensor(paddle::experimental::reshape(self._PD_GetInner(), new_shape));
}

inline at::Tensor unflatten_symint(const at::Tensor& self,
                                   const int64_t dim,
                                   c10::SymIntArrayRef sizes) {
  return unflatten(
      self,
      dim,
      at::IntArrayRef(reinterpret_cast<const int64_t*>(sizes.data()),
                      sizes.size()));
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::unflatten(int64_t dim, at::IntArrayRef sizes) const {
  return at::unflatten(*this, dim, sizes);
}

inline at::Tensor Tensor::unflatten_symint(int64_t dim,
                                           c10::SymIntArrayRef sizes) const {
  return at::unflatten_symint(*this, dim, sizes);
}

}  // namespace at
