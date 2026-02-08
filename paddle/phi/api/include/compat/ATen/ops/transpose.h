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
#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"

namespace at {

inline at::Tensor transpose(const at::Tensor& self,
                            const at::Scalar& dim0,
                            const at::Scalar& dim1) {
  int d0 = dim0.to<int>();
  int d1 = dim1.to<int>();
  int64_t ndim = self.dim();

  if (d0 < 0) d0 += ndim;
  if (d1 < 0) d1 += ndim;

  PD_CHECK(d0 >= 0 && d0 < ndim, "dim0 out of range");
  PD_CHECK(d1 >= 0 && d1 < ndim, "dim1 out of range");

  std::vector<int> perm(ndim);
  for (int i = 0; i < ndim; ++i) {
    perm[i] = i;
  }
  std::swap(perm[d0], perm[d1]);

  return paddle::experimental::transpose(self._PD_GetInner(), perm);
}

}  // namespace at

namespace torch {
using at::transpose;
}  // namespace torch
