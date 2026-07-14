// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <c10/core/TensorOptions.h>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

#include "paddle/phi/api/include/api.h"

namespace at::detail {

inline int _PD_normalize_transpose_dim(int64_t dim,
                                       int64_t ndim,
                                       const char* name) {
  int64_t normalized = dim;
  if (normalized < 0) {
    normalized += ndim;
  }

  PD_CHECK(normalized >= 0 && normalized < ndim, name, " out of range");
  PD_CHECK(normalized <= static_cast<int64_t>(std::numeric_limits<int>::max()),
           name,
           " out of int range");
  return static_cast<int>(normalized);
}

inline std::vector<int> _PD_make_transpose_perm(int64_t ndim, int d0, int d1) {
  PD_CHECK(ndim <= static_cast<int64_t>(std::numeric_limits<int>::max()),
           "tensor rank out of int range");

  std::vector<int> perm(static_cast<size_t>(ndim));
  for (int64_t i = 0; i < ndim; ++i) {
    perm[static_cast<size_t>(i)] = static_cast<int>(i);
  }
  std::swap(perm[d0], perm[d1]);
  return perm;
}

}  // namespace at::detail

namespace at {

inline at::Tensor transpose(const at::Tensor& self,
                            int64_t dim0,
                            int64_t dim1) {
  int64_t ndim = self.dim();
  int d0 = at::detail::_PD_normalize_transpose_dim(dim0, ndim, "dim0");
  int d1 = at::detail::_PD_normalize_transpose_dim(dim1, ndim, "dim1");
  auto perm = at::detail::_PD_make_transpose_perm(ndim, d0, d1);

  return paddle::experimental::transpose(self._PD_GetInner(), perm);
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  return at::transpose(*this, dim0, dim1);
}

inline at::Tensor& Tensor::transpose_(int64_t dim0, int64_t dim1) const {
  int64_t ndim = this->dim();
  int d0 = at::detail::_PD_normalize_transpose_dim(dim0, ndim, "dim0");
  int d1 = at::detail::_PD_normalize_transpose_dim(dim1, ndim, "dim1");
  auto perm = at::detail::_PD_make_transpose_perm(ndim, d0, d1);
  PaddleTensor& inner = const_cast<PaddleTensor&>(tensor_);
  paddle::experimental::transpose_(inner, perm);
  return const_cast<at::Tensor&>(*this);
}

}  // namespace at
