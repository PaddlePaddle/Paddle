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
#include <ATen/ops/split.h>

namespace at {

inline std::vector<at::Tensor> unsafe_split(const at::Tensor& self,
                                            int64_t split_size,
                                            int64_t dim) {
  return at::split(self, split_size, dim);
}

inline std::vector<at::Tensor> unsafe_split_symint(const at::Tensor& self,
                                                   c10::SymInt split_size,
                                                   int64_t dim) {
  return at::split(self, static_cast<int64_t>(split_size), dim);
}

}  // namespace at

namespace at {

inline std::vector<at::Tensor> Tensor::unsafe_split(int64_t split_size,
                                                    int64_t dim) const {
  return at::unsafe_split(*this, split_size, dim);
}

inline std::vector<at::Tensor> Tensor::unsafe_split_symint(
    c10::SymInt split_size, int64_t dim) const {
  return at::unsafe_split_symint(*this, split_size, dim);
}

}  // namespace at
