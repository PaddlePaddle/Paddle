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
#include <c10/core/List.h>
#include <c10/core/Scalar.h>

#include "paddle/phi/api/include/api.h"

namespace at {

// Helper function to convert c10::List<optional<Tensor>> to vector<Tensor>
inline std::vector<at::Tensor> convert_indices_list(
    const c10::List<::std::optional<at::Tensor>>& indices) {
  std::vector<at::Tensor> result;
  result.reserve(indices.size());
  for (const auto& idx : indices) {
    if (idx.has_value()) {
      result.push_back(idx.value());
    }
  }
  return result;
}

// index: Get values at specified indices
inline at::Tensor index(const at::Tensor& self,
                        const c10::List<::std::optional<at::Tensor>>& indices) {
  std::vector<paddle::Tensor> pd_indices;
  pd_indices.reserve(indices.size());

  // Convert indices list to vector, filtering out None values
  for (const auto& idx : indices) {
    if (idx.has_value()) {
      pd_indices.push_back(idx.value()._PD_GetInner());
    }
  }

  // Handle empty indices - return self
  if (pd_indices.empty()) {
    return self;
  }

  // Simple case: single index tensor
  if (pd_indices.size() == 1 && indices.size() == 1) {
    // Use index_select on dimension 0
    return paddle::experimental::index_select(
        self._PD_GetInner(), pd_indices[0], 0);
  }

  // Multi-dimensional indexing using gather_nd when all indices are present
  if (pd_indices.size() == indices.size()) {
    // Stack indices along last dimension for gather_nd
    // gather_nd expects shape [batch, ..., num_indices]
    auto stacked_indices = paddle::experimental::stack(pd_indices, -1);
    return paddle::experimental::gather_nd(self._PD_GetInner(),
                                           stacked_indices);
  }

  // Complex case with mixed None and Tensor indices
  // Requires more sophisticated handling
  // For now, we use index_elementwise_get which is more flexible

  // Calculate strides and dimensions
  auto self_dims = self._PD_GetInner().dims();
  std::vector<int64_t> input_dims;
  std::vector<int64_t> input_strides;

  // Build input dimensions and strides
  int64_t stride = 1;
  for (int64_t i = self_dims.size() - 1; i >= 0; --i) {
    input_dims.insert(input_dims.begin(), self_dims[i]);
    input_strides.insert(input_strides.begin(), stride);
    stride *= self_dims[i];
  }

  // For index tensors, we need to calculate their dimensions and strides
  if (!pd_indices.empty()) {
    auto idx_dims = pd_indices[0].dims();
    std::vector<int64_t> index_dims;
    std::vector<int64_t> index_strides_vec;

    int64_t idx_stride = 1;
    for (int64_t i = idx_dims.size() - 1; i >= 0; --i) {
      index_dims.insert(index_dims.begin(), idx_dims[i]);
      index_strides_vec.insert(index_strides_vec.begin(), idx_stride);
      idx_stride *= idx_dims[i];
    }

    // Use index_elementwise_get for general case
    return paddle::experimental::index_elementwise_get(self._PD_GetInner(),
                                                       pd_indices,
                                                       input_dims,
                                                       input_strides,
                                                       index_dims,
                                                       index_strides_vec,
                                                       /*slice_offset=*/0,
                                                       /*accumulate=*/false,
                                                       /*is_combined=*/false);
  }

  return self;
}

// index_put_: Set values at specified indices (in-place)
inline at::Tensor& index_put_(
    at::Tensor& self,  // NOLINT(runtime/references)
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate) {
  std::vector<paddle::Tensor> pd_indices;
  pd_indices.reserve(indices.size());
  for (const auto& idx : indices) {
    if (idx.has_value()) {
      pd_indices.push_back(idx.value()._PD_GetInner());
    }
  }

  paddle::experimental::index_put_(
      self._PD_GetInner(), pd_indices, values._PD_GetInner(), accumulate);
  return self;
}

// index_put_: Set scalar value at specified indices (in-place)
inline at::Tensor& index_put_(
    at::Tensor& self,  // NOLINT(runtime/references)
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Scalar& v,
    bool accumulate) {
  std::vector<paddle::Tensor> pd_indices;
  pd_indices.reserve(indices.size());
  for (const auto& idx : indices) {
    if (idx.has_value()) {
      pd_indices.push_back(idx.value()._PD_GetInner());
    }
  }

  // Create a scalar tensor from the value
  auto scalar_tensor = paddle::experimental::full(
      {}, phi::Scalar(v.to<double>()), self._PD_GetInner().dtype());

  paddle::experimental::index_put_(
      self._PD_GetInner(), pd_indices, scalar_tensor, accumulate);
  return self;
}

// index_put: Non-inplace version
inline at::Tensor index_put(
    const at::Tensor& self,
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate) {
  std::vector<paddle::Tensor> pd_indices;
  pd_indices.reserve(indices.size());
  for (const auto& idx : indices) {
    if (idx.has_value()) {
      pd_indices.push_back(idx.value()._PD_GetInner());
    }
  }

  return paddle::experimental::index_put(
      self._PD_GetInner(), pd_indices, values._PD_GetInner(), accumulate);
}

// Tensor member function implementations
inline at::Tensor Tensor::index(
    const c10::List<::std::optional<at::Tensor>>& indices) const {
  return at::index(*this, indices);
}

inline at::Tensor& Tensor::index_put_(
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate) const {
  return at::index_put_(
      const_cast<at::Tensor&>(*this), indices, values, accumulate);
}

inline at::Tensor& Tensor::index_put_(
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Scalar& v,
    bool accumulate) const {
  return at::index_put_(const_cast<at::Tensor&>(*this), indices, v, accumulate);
}

inline at::Tensor Tensor::index_put(
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate) const {
  return at::index_put(*this, indices, values, accumulate);
}

}  // namespace at

namespace torch {
using at::index;
using at::index_put;
using at::index_put_;
}  // namespace torch
