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

inline std::vector<at::Tensor> split(const at::Tensor& self,
                                     int64_t split_size,
                                     int64_t dim = 0) {
  // Calculate number of splits based on split_size
  int64_t dim_size = self._PD_GetInner().dims()[dim];
  std::vector<int64_t> split_sizes;
  for (int64_t i = 0; i < dim_size; i += split_size) {
    split_sizes.push_back(std::min(split_size, dim_size - i));
  }
  auto outputs =
      paddle::experimental::split(self._PD_GetInner(), split_sizes, dim);
  std::vector<at::Tensor> at_tensors;
  at_tensors.reserve(outputs.size());
  for (const auto& paddle_tensor : outputs) {
    at_tensors.emplace_back(paddle_tensor);
  }
  return at_tensors;
}

inline std::vector<at::Tensor> split_symint(const at::Tensor& self,
                                            c10::SymInt split_size,
                                            int64_t dim = 0) {
  return split(self, static_cast<int64_t>(split_size), dim);
}

inline std::vector<at::Tensor> split(const at::Tensor& self,
                                     at::IntArrayRef split_sizes,
                                     int64_t dim = 0) {
  auto outputs = paddle::experimental::split(
      self._PD_GetInner(), split_sizes._PD_ToPaddleIntArray(), dim);
  std::vector<at::Tensor> at_tensors;
  at_tensors.reserve(outputs.size());
  for (const auto& paddle_tensor : outputs) {
    at_tensors.emplace_back(paddle_tensor);
  }
  return at_tensors;
}

inline std::vector<at::Tensor> split_symint(const at::Tensor& self,
                                            c10::SymIntArrayRef split_sizes,
                                            int64_t dim = 0) {
  return split(
      self,
      at::IntArrayRef(reinterpret_cast<const int64_t*>(split_sizes.data()),
                      split_sizes.size()),
      dim);
}

}  // namespace at

namespace at {

inline std::vector<at::Tensor> Tensor::split(int64_t split_size,
                                             int64_t dim = 0) const {
  return at::split(*this, split_size, dim);
}

inline std::vector<at::Tensor> Tensor::split_symint(c10::SymInt split_size,
                                                    int64_t dim = 0) const {
  return at::split_symint(*this, split_size, dim);
}

inline std::vector<at::Tensor> Tensor::split(at::IntArrayRef split_sizes,
                                             int64_t dim = 0) const {
  return at::split(*this, split_sizes, dim);
}

inline std::vector<at::Tensor> Tensor::split_symint(
    c10::SymIntArrayRef split_sizes, int64_t dim = 0) const {
  return at::split_symint(*this, split_sizes, dim);
}

}  // namespace at
