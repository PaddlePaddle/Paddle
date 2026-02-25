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
// PyTorch's expand works by right-aligning dimensions and broadcasting
// dimensions with size 1 to the target size
inline Tensor expand(const Tensor& self,
                     at::IntArrayRef size,
                     bool implicit = false) {
  // implicit parameter is used by PyTorch's vmap for internal optimization.
  // It doesn't affect the actual expand operation, so we can safely ignore it.

  paddle::Tensor pd_tensor = self._PD_GetInner();

  // Target sizes - convert to vector
  std::vector<int64_t> target_size_vec(size.begin(), size.end());

  // Use Paddle's native expand API
  paddle::Tensor result = pd_tensor.expand(phi::IntArray(target_size_vec));

  return Tensor(result);
}

// expand_as - expands to same size as another tensor
inline Tensor expand_as(const Tensor& self, const Tensor& other) {
  return expand(self, other.sizes());
}

}  // namespace at

namespace at {

// Member function: Tensor::expand
inline Tensor Tensor::expand(at::IntArrayRef size, bool implicit) const {
  return at::expand(*this, size, implicit);
}

// Member function: Tensor::expand_as
inline Tensor Tensor::expand_as(const Tensor& other) const {
  return at::expand_as(*this, other);
}

}  // namespace at
