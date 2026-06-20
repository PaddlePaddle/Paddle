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
#include <ATen/ops/index.h>
#include <c10/core/List.h>
#include <c10/core/Scalar.h>

#include "paddle/phi/api/include/api.h"

namespace at::detail {

inline std::vector<at::Tensor> _PD_convert_indices_list(
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

inline c10::List<::std::optional<at::Tensor>> _PD_convert_tensor_index_list(
    ArrayRef<at::indexing::TensorIndex> indices) {
  c10::List<::std::optional<at::Tensor>> result;
  for (const auto& index : indices) {
    PD_CHECK(!index.is_ellipsis(), "Ellipsis index is not supported yet.");
    PD_CHECK(!index.is_integer(), "Integer index is not supported yet.");
    PD_CHECK(!index.is_boolean(), "Boolean index is not supported yet.");
    PD_CHECK(!index.is_slice(), "Slice index is not supported yet.");
    if (index.is_none()) {
      result.push_back(::std::nullopt);
    } else if (index.is_tensor()) {
      result.push_back(index.tensor());
    }
  }
  return result;
}

}  // namespace at::detail

namespace at {

// index_put_: Set values at specified indices (in-place)
inline at::Tensor& index_put_(
    at::Tensor& self,  // NOLINT(runtime/references)
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate = false) {
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

// index_put: Non-inplace version
inline at::Tensor index_put(
    const at::Tensor& self,
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate = false) {
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
    ArrayRef<at::indexing::TensorIndex> indices, Tensor const& rhs) {
  return index_put_(detail::_PD_convert_tensor_index_list(indices), rhs);
}

inline at::Tensor& Tensor::index_put_(
    ArrayRef<at::indexing::TensorIndex> indices, const Scalar& v) {
  auto scalar_tensor = at::Tensor(paddle::experimental::full(
      {}, phi::Scalar(v.to<double>()), this->_PD_GetInner().dtype()));
  return index_put_(indices, scalar_tensor);
}

inline at::Tensor& Tensor::index_put_(
    std::initializer_list<at::indexing::TensorIndex> indices,
    Tensor const& rhs) {
  return index_put_(ArrayRef<at::indexing::TensorIndex>(indices), rhs);
}

inline at::Tensor& Tensor::index_put_(
    std::initializer_list<at::indexing::TensorIndex> indices, const Scalar& v) {
  return index_put_(ArrayRef<at::indexing::TensorIndex>(indices), v);
}

inline at::Tensor Tensor::index_put(
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate) const {
  return at::index_put(*this, indices, values, accumulate);
}

}  // namespace at
