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

inline std::vector<at::Tensor> unsafe_split_with_sizes(
    const at::Tensor& self, at::IntArrayRef split_sizes, int64_t dim = 0) {
  return at::split(self, split_sizes, dim);
}

inline std::vector<at::Tensor> unsafe_split_with_sizes_symint(
    const at::Tensor& self, c10::SymIntArrayRef split_sizes, int64_t dim = 0) {
  return at::split_symint(self, split_sizes, dim);
}

}  // namespace at

namespace at {

inline std::vector<at::Tensor> Tensor::unsafe_split_with_sizes(
    at::IntArrayRef split_sizes, int64_t dim) const {
  return at::unsafe_split_with_sizes(*this, split_sizes, dim);
}

inline std::vector<at::Tensor> Tensor::unsafe_split_with_sizes_symint(
    c10::SymIntArrayRef split_sizes, int64_t dim) const {
  return at::unsafe_split_with_sizes_symint(*this, split_sizes, dim);
}

}  // namespace at
