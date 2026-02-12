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

inline at::Tensor permute(const at::Tensor& self, at::IntArrayRef dims) {
  std::vector<int> perm(dims.size());
  for (size_t i = 0; i < dims.size(); i++) {
    perm[i] = static_cast<int>(dims[i]);
  }
  return paddle::experimental::transpose(self._PD_GetInner(), perm);
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::permute(at::IntArrayRef dims) const {
  return at::permute(*this, dims);
}

}  // namespace at
