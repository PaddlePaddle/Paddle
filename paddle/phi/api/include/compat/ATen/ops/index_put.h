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
#include <vector>

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
    if (index.is_slice()) {
      PD_CHECK(_PD_is_full_slice(index.slice()),
               "Only full Slice() is supported in index_put_ TensorIndex "
               "paths.");
    } else if (index.is_tensor()) {
      result.push_back(index.tensor());
    }
  }
  return result;
}

inline at::Tensor _PD_squeeze_newaxis_value(
    const at::Tensor& values, ArrayRef<at::indexing::TensorIndex> indices) {
  std::vector<int64_t> value_shape(values.sizes().begin(),
                                   values.sizes().end());
  size_t value_dim = 0;
  bool changed = false;

  for (const auto& index : indices) {
    if (index.is_none()) {
      if (!value_shape.empty()) {
        PD_CHECK(value_dim < value_shape.size(),
                 "index_put_ value rank is too small for None index.");
        PD_CHECK(value_shape[value_dim] == 1,
                 "index_put_ expected value dimension inserted by None to "
                 "have size 1, but got ",
                 value_shape[value_dim],
                 ".");
        value_shape.erase(value_shape.begin() + value_dim);
        changed = true;
      }
    } else if (index.is_tensor()) {
      if (!value_shape.empty()) {
        ++value_dim;
      }
    } else if (index.is_slice()) {
      PD_CHECK(_PD_is_full_slice(index.slice()),
               "Only full Slice() is supported in index_put_ TensorIndex "
               "paths.");
      if (!value_shape.empty()) {
        ++value_dim;
      }
    } else {
      PD_CHECK(!index.is_ellipsis(), "Ellipsis index is not supported yet.");
      PD_CHECK(!index.is_integer(), "Integer index is not supported yet.");
      PD_CHECK(!index.is_boolean(), "Boolean index is not supported yet.");
    }
  }

  if (!changed) {
    return values;
  }
  return paddle::experimental::reshape(values._PD_GetInner(),
                                       phi::IntArray(value_shape));
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
  auto tensor_indices = detail::_PD_convert_tensor_index_list(indices);
  at::Tensor values = detail::_PD_squeeze_newaxis_value(rhs, indices);
  if (tensor_indices.empty()) {
    return copy_(values);
  }
  return index_put_(tensor_indices, values);
}

inline at::Tensor& Tensor::index_put_(
    ArrayRef<at::indexing::TensorIndex> indices, const Scalar& v) {
  auto tensor_indices = detail::_PD_convert_tensor_index_list(indices);
  if (tensor_indices.empty()) {
    std::vector<int64_t> value_shape(this->sizes().begin(),
                                     this->sizes().end());
    auto scalar_tensor =
        at::Tensor(paddle::experimental::full(phi::IntArray(value_shape),
                                              phi::Scalar(v.to<double>()),
                                              this->_PD_GetInner().dtype()));
    return copy_(scalar_tensor);
  }
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
