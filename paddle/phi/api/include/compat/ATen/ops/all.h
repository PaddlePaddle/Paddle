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
#include <c10/util/OptionalArrayRef.h>

#include "paddle/phi/api/include/api.h"

namespace at {

// all: Check if all elements are true (non-zero)
// Version 1: all() - check all elements in the tensor
inline at::Tensor all(const at::Tensor& self) {
  return paddle::experimental::all(self._PD_GetInner(), {}, false);
}

// Version 2: all(dim, keepdim) - check along a specific dimension
inline at::Tensor all(const at::Tensor& self,
                      int64_t dim,
                      bool keepdim = false) {
  return paddle::experimental::all(self._PD_GetInner(), {dim}, keepdim);
}

// Version 3: all(dim, keepdim) - check along optional dimensions
inline at::Tensor all(const at::Tensor& self,
                      at::OptionalIntArrayRef dim,
                      bool keepdim = false) {
  std::vector<int64_t> axis_vec;
  if (dim.has_value()) {
    axis_vec.assign(dim.value().begin(), dim.value().end());
  }
  return paddle::experimental::all(self._PD_GetInner(), axis_vec, keepdim);
}

}  // namespace at

namespace at {

// Tensor member function implementations
inline at::Tensor Tensor::all() const { return at::all(*this); }

inline at::Tensor Tensor::all(int64_t dim, bool keepdim) const {
  return at::all(*this, dim, keepdim);
}

inline at::Tensor Tensor::all(at::OptionalIntArrayRef dim, bool keepdim) const {
  return at::all(*this, dim, keepdim);
}

}  // namespace at
