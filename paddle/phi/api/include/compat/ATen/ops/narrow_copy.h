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
#include <ATen/ops/narrow.h>

namespace at {

inline at::Tensor narrow_copy(const at::Tensor& self,
                              int64_t dim,
                              int64_t start,
                              int64_t length) {
  // narrow_copy returns a copy of the narrowed tensor
  return narrow(self, dim, start, length).clone();
}

inline at::Tensor narrow_copy_symint(const at::Tensor& self,
                                     int64_t dim,
                                     c10::SymInt start,
                                     c10::SymInt length) {
  return narrow_copy(self, dim, start, length);
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::narrow_copy(int64_t dim,
                                      int64_t start,
                                      int64_t length) const {
  return at::narrow_copy(*this, dim, start, length);
}

inline at::Tensor Tensor::narrow_copy_symint(int64_t dim,
                                             c10::SymInt start,
                                             c10::SymInt length) const {
  return at::narrow_copy_symint(*this, dim, start, length);
}

}  // namespace at
