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

inline at::Tensor slice(const at::Tensor& self,
                        int64_t dim = 0,
                        ::std::optional<int64_t> start = ::std::nullopt,
                        ::std::optional<int64_t> end = ::std::nullopt,
                        int64_t step = 1) {
  // Materialize the compat StorageHolderView before creating the slice so the
  // base tensor and its views observe the same shared storage during resize_.
  (void)self.storage();
  return paddle::experimental::slice(
      self._PD_GetInner(),
      {dim},
      start.has_value() ? IntArrayRef(start.value())._PD_ToPaddleIntArray()
                        : IntArrayRef()._PD_ToPaddleIntArray(),
      end.has_value() ? IntArrayRef(end.value())._PD_ToPaddleIntArray()
                      : IntArrayRef()._PD_ToPaddleIntArray(),
      {1},
      {});
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::slice(int64_t dim,
                                ::std::optional<int64_t> start,
                                ::std::optional<int64_t> end,
                                int64_t step) {
  return at::slice(*this, dim, start, end, step);
}

}  // namespace at
