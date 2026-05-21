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
  // Handle empty indices - return self
  if (indices.empty()) {
    return self;
  }

  // Check if all indices are None (select all)
  bool all_none = true;
  for (const auto& idx : indices) {
    if (idx.has_value()) {
      all_none = false;
      break;
    }
  }
  if (all_none) {
    return self;
  }

  // Build vector of indices while tracking which dimensions have None
  std::vector<paddle::Tensor> pd_indices;
  std::vector<bool> has_index(indices.size(), false);
  pd_indices.reserve(indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i].has_value()) {
      pd_indices.push_back(indices[i].value()._PD_GetInner());
      has_index[i] = true;
    } else {
      // None - will be handled as "select all"
      pd_indices.push_back(paddle::Tensor());
      has_index[i] = false;
    }
  }

  // If only one non-None index at position 0, use index_select
  int non_none_count = 0;
  size_t first_non_none = 0;
  for (size_t i = 0; i < has_index.size(); ++i) {
    if (has_index[i]) {
      non_none_count++;
      first_non_none = i;
    }
  }

  // Simple case: single index tensor
  if (non_none_count == 1 && first_non_none == 0) {
    return paddle::experimental::index_select(
        self._PD_GetInner(), pd_indices[0], 0);
  }

  // Case: All indices are tensors (no None) - use gather_nd
  if (non_none_count == static_cast<int>(indices.size())) {
    auto stacked_indices = paddle::experimental::stack(pd_indices, -1);
    return paddle::experimental::gather_nd(self._PD_GetInner(),
                                           stacked_indices);
  }

  // Mixed case: some indices are None (select all) and some are tensors
  // Handle by using a combination of operations
  auto self_dims = self._PD_GetInner().dims();
  int self_rank = self_dims.size();

  // Build result by iterating over dimensions
  // For dimensions with None, we select all; for dimensions with tensor, we use
  // the index
  at::Tensor result = self;

  for (size_t i = 0; i < indices.size() && i < static_cast<size_t>(self_rank);
       ++i) {
    if (has_index[i]) {
      // Use the index tensor for this dimension
      paddle::Tensor pd_result = result._PD_GetInner();
      result = paddle::experimental::index_select(
          pd_result, pd_indices[i], static_cast<int>(i));
    }
    // If None, we select all along this dimension (do nothing)
  }

  return result;
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

}  // namespace at

namespace at {

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
