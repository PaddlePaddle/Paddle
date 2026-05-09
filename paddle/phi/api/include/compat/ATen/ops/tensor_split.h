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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <ATen/core/Tensor.h>

namespace at {

inline std::vector<at::Tensor> tensor_split(const at::Tensor& self,
                                            int64_t sections,
                                            int64_t dim = 0) {
  // Follow PyTorch's tensor_split_sections_symint implementation
  PD_CHECK(self._PD_GetInner().dims().size() > 0,
           "tensor_split expected at least a 1-dimensional tensor, but got a "
           "tensor with ",
           self._PD_GetInner().dims().size(),
           " dims");

  PD_CHECK(
      sections > 0, "number of sections must be larger than 0, got ", sections);

  int64_t dim_size = self._PD_GetInner().dims()[dim];

  // Calculate split sizes: first (dim_size % sections) chunks get size
  // (dim_size / sections + 1), remaining chunks get size (dim_size / sections)
  auto min_split_size = dim_size / sections;
  auto num_splits_one_extra = dim_size % sections;

  std::vector<int64_t> split_sizes;
  split_sizes.reserve(sections);

  for (int64_t split_idx = 0; split_idx < sections; ++split_idx) {
    auto split_size = (split_idx < num_splits_one_extra) ? (min_split_size + 1)
                                                         : min_split_size;
    split_sizes.push_back(split_size);
  }

  // Use split with calculated sizes
  auto outputs =
      paddle::experimental::split(self._PD_GetInner(), split_sizes, dim);

  std::vector<at::Tensor> at_tensors;
  at_tensors.reserve(outputs.size());
  for (const auto& paddle_tensor : outputs) {
    at_tensors.emplace_back(paddle_tensor);
  }
  return at_tensors;
}

inline std::vector<at::Tensor> tensor_split_symint(const at::Tensor& self,
                                                   c10::SymInt sections,
                                                   int64_t dim = 0) {
  return tensor_split(self, static_cast<int64_t>(sections), dim);
}

inline std::vector<at::Tensor> tensor_split(const at::Tensor& self,
                                            at::IntArrayRef indices,
                                            int64_t dim = 0) {
  // Follow PyTorch's _tensor_split_indices implementation
  // indices are split positions, not sizes
  PD_CHECK(self._PD_GetInner().dims().size() > 0,
           "tensor_split expected at least a 1-dimensional tensor, but got a "
           "tensor with ",
           self._PD_GetInner().dims().size(),
           " dims");

  int64_t num_indices = indices.size();
  int64_t dim_size = self._PD_GetInner().dims()[dim];

  // Convert indices (positions) to sizes
  std::vector<int64_t> split_sizes;
  split_sizes.reserve(num_indices + 1);

  int64_t start_idx = 0;
  for (int64_t i = 0; i < num_indices; ++i) {
    int64_t end_idx = indices[i];
    // Handle negative indices
    if (end_idx < 0) {
      end_idx += dim_size;
    }
    // Clamp to valid range
    end_idx = std::max(start_idx, std::min(end_idx, dim_size));
    split_sizes.push_back(end_idx - start_idx);
    start_idx = end_idx;
  }
  // Add the last segment
  split_sizes.push_back(dim_size - start_idx);

  // Use split with calculated sizes
  auto outputs =
      paddle::experimental::split(self._PD_GetInner(), split_sizes, dim);

  std::vector<at::Tensor> at_tensors;
  at_tensors.reserve(outputs.size());
  for (const auto& paddle_tensor : outputs) {
    at_tensors.emplace_back(paddle_tensor);
  }
  return at_tensors;
}

inline std::vector<at::Tensor> tensor_split_symint(const at::Tensor& self,
                                                   c10::SymIntArrayRef indices,
                                                   int64_t dim = 0) {
  return tensor_split(
      self,
      at::IntArrayRef(reinterpret_cast<const int64_t*>(indices.data()),
                      indices.size()),
      dim);
}

inline std::vector<at::Tensor> tensor_split(
    const at::Tensor& self,
    const at::Tensor& tensor_indices_or_sections,
    int64_t dim = 0) {
  // Follow PyTorch's validation and implementation
  PD_CHECK(self._PD_GetInner().dims().size() > 0,
           "tensor_split expected at least a 1-dimensional tensor, but got a "
           "tensor with ",
           self._PD_GetInner().dims().size(),
           " dims");

  auto split_device = tensor_indices_or_sections.device();
  PD_CHECK(split_device.is_cpu(),
           "tensor_split expected tensor_indices_or_sections to be on cpu, but "
           "it's on ",
           split_device);

  auto split_dtype = tensor_indices_or_sections.scalar_type();
  PD_CHECK(split_dtype == at::kLong,
           "tensor_split expected tensor_indices_or_sections to have dtype of "
           "long, but got ",
           split_dtype);

  auto split_dim = tensor_indices_or_sections.dim();
  PD_CHECK(split_dim == 1 || split_dim == 0,
           "tensor_split expected tensor_indices_or_sections to be a "
           "zero-dimensional or one-dimensional tensor, but got a tensor with ",
           split_dim,
           " dims");

  if (split_dim == 0) {
    // 0-dimensional tensor: treat as sections
    int64_t sections = tensor_indices_or_sections.item<int64_t>();
    return tensor_split(self, sections, dim);
  } else {
    // 1-dimensional tensor: treat as indices
    // Need to handle non-contiguous tensors properly
    const PaddleTensor& paddle_tensor =
        tensor_indices_or_sections._PD_GetInner();
    const int64_t* indices_data = paddle_tensor.data<int64_t>();
    auto stride = tensor_indices_or_sections.stride(0);
    auto numel = tensor_indices_or_sections.numel();
    std::vector<int64_t> indices(numel);
    for (int64_t offset = 0; offset < numel; ++offset) {
      // indices tensor could be non-contiguous
      indices[offset] = *(indices_data + offset * stride);
    }
    return tensor_split(self, at::IntArrayRef(indices), dim);
  }
}

}  // namespace at

namespace at {

inline std::vector<at::Tensor> Tensor::tensor_split(int64_t sections,
                                                    int64_t dim = 0) const {
  return at::tensor_split(*this, sections, dim);
}

inline std::vector<at::Tensor> Tensor::tensor_split_symint(
    c10::SymInt sections, int64_t dim = 0) const {
  return at::tensor_split_symint(*this, sections, dim);
}

inline std::vector<at::Tensor> Tensor::tensor_split(at::IntArrayRef indices,
                                                    int64_t dim = 0) const {
  return at::tensor_split(*this, indices, dim);
}

inline std::vector<at::Tensor> Tensor::tensor_split_symint(
    c10::SymIntArrayRef indices, int64_t dim = 0) const {
  return at::tensor_split_symint(*this, indices, dim);
}

inline std::vector<at::Tensor> Tensor::tensor_split(
    const at::Tensor& tensor_indices_or_sections, int64_t dim = 0) const {
  return at::tensor_split(*this, tensor_indices_or_sections, dim);
}

}  // namespace at
