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
#include <c10/core/Scalar.h>
#include <c10/util/OptionalArrayRef.h>

#include "paddle/phi/api/include/api.h"

namespace at {

// any - free functions
inline Tensor any(const Tensor& self, int64_t dim, bool keepdim = false) {
  return paddle::experimental::any(self._PD_GetInner(), {dim}, keepdim);
}

inline Tensor any(const Tensor& self,
                  at::OptionalIntArrayRef dim,
                  bool keepdim = false) {
  std::vector<int64_t> dims_vec;
  if (dim.has_value() && dim.value().size() > 0) {
    dims_vec.assign(dim.value().begin(), dim.value().end());
  }
  return paddle::experimental::any(self._PD_GetInner(), dims_vec, keepdim);
}

inline Tensor any(const Tensor& self) {
  return paddle::experimental::any(self._PD_GetInner());
}

// any - member function implementations
inline Tensor Tensor::any(int64_t dim, bool keepdim) const {
  return paddle::experimental::any(_PD_GetInner(), {dim}, keepdim);
}

inline Tensor Tensor::any(at::OptionalIntArrayRef dim, bool keepdim) const {
  std::vector<int64_t> dims_vec;
  if (dim.has_value() && dim.value().size() > 0) {
    dims_vec.assign(dim.value().begin(), dim.value().end());
  }
  return paddle::experimental::any(_PD_GetInner(), dims_vec, keepdim);
}

inline Tensor Tensor::any() const {
  return paddle::experimental::any(_PD_GetInner());
}

}  // namespace at
