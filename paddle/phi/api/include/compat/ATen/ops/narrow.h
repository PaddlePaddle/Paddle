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

inline at::Tensor narrow(const at::Tensor& self,
                         int64_t dim,
                         int64_t start,
                         int64_t length) {
  // Use slice to implement narrow: narrow(dim, start, length) is equivalent
  // to slice(dim, start, start + length)
  return Tensor(paddle::experimental::slice(
      self._PD_GetInner(), {dim}, {start}, {start + length}, {1}, {}));
}

inline at::Tensor narrow_symint(const at::Tensor& self,
                                int64_t dim,
                                c10::SymInt start,
                                c10::SymInt length) {
  return narrow(self, dim, start, length);
}

inline at::Tensor narrow(const at::Tensor& self,
                         int64_t dim,
                         const at::Tensor& start,
                         int64_t length) {
  // Extract scalar value from start tensor
  PD_CHECK(start.numel() == 1,
           "start must be a 0-dim tensor or 1-element tensor");
  int64_t start_val = start.item<int64_t>();
  return narrow(self, dim, start_val, length);
}

inline at::Tensor narrow_symint(const at::Tensor& self,
                                int64_t dim,
                                const at::Tensor& start,
                                c10::SymInt length) {
  return narrow(self, dim, start, length);
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::narrow(int64_t dim,
                                 int64_t start,
                                 int64_t length) const {
  return at::narrow(*this, dim, start, length);
}

inline at::Tensor Tensor::narrow_symint(int64_t dim,
                                        c10::SymInt start,
                                        c10::SymInt length) const {
  return at::narrow_symint(*this, dim, start, length);
}

inline at::Tensor Tensor::narrow(int64_t dim,
                                 const at::Tensor& start,
                                 int64_t length) const {
  return at::narrow(*this, dim, start, length);
}

inline at::Tensor Tensor::narrow_symint(int64_t dim,
                                        const at::Tensor& start,
                                        c10::SymInt length) const {
  return at::narrow_symint(*this, dim, start, length);
}

}  // namespace at
