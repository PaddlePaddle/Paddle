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
#include <ATen/core/TensorBase.h>
#include <ATen/ops/item.h>
#include <c10/core/ScalarType.h>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "paddle/phi/api/include/api.h"

namespace at {

namespace detail {

// Validate that int64_t dim fits in int before static_cast.
inline int _pd_check_dim(int64_t dim) {
  if (dim < INT_MIN || dim > INT_MAX) {
    throw std::out_of_range("scatter_reduce: dim is out of int range.");
  }
  return static_cast<int>(dim);
}

inline int _pd_normalize_dim(int64_t dim, int64_t rank) {
  if (dim < -rank || dim >= rank) {
    throw std::out_of_range("scatter_reduce: dim is out of range.");
  }
  if (dim < 0) {
    dim += rank;
  }
  return _pd_check_dim(dim);
}

inline void _pd_check_index(const at::Tensor& index) {
  if (index.numel() == 0) {
    return;
  }
  auto index_type = index.scalar_type();
  if (index_type != at::kLong && index_type != at::kInt) {
    throw std::invalid_argument(
        "scatter_reduce: index must have int32 or int64 dtype.");
  }
  auto min_index = paddle::experimental::min(index._PD_GetInner());
  auto min_index_tensor = at::Tensor(std::move(min_index));
  int64_t min_index_value =
      index_type == at::kLong
          ? min_index_tensor.item<int64_t>()
          : static_cast<int64_t>(min_index_tensor.item<int>());
  if (min_index_value < 0) {
    throw std::out_of_range("scatter_reduce: index contains negative values.");
  }
}

inline void _pd_check_scatter_reduce_shape(const at::Tensor& self,
                                           const at::Tensor& index,
                                           const at::Tensor& src,
                                           int64_t normalized_dim) {
  PD_CHECK(index.dim() == src.dim(),
           "scatter_reduce: index and src must have the same rank.");
  PD_CHECK(index.dim() == self.dim(),
           "scatter_reduce: index and self must have the same rank.");
  for (int64_t i = 0; i < index.dim(); ++i) {
    PD_CHECK(index.size(i) <= src.size(i),
             "scatter_reduce: index.size(",
             i,
             ") must be no larger than src.size(",
             i,
             ").");
    if (i != normalized_dim) {
      PD_CHECK(index.size(i) <= self.size(i),
               "scatter_reduce: index.size(",
               i,
               ") must be no larger than self.size(",
               i,
               ") for dimensions other than dim.");
    }
  }
}

// Convert PyTorch scatter_reduce reduce string to Paddle put_along_axis
// reduce string.
inline std::string _pd_scatter_reduce_str(c10::string_view reduce) {
  if (reduce == "sum") {
    return "add";
  } else if (reduce == "prod") {
    return "multiply";
  } else if (reduce == "mean") {
    return "mean";
  } else if (reduce == "amax") {
    return "amax";
  } else if (reduce == "amin") {
    return "amin";
  }
  throw std::invalid_argument(
      "scatter_reduce: unsupported reduce mode '" + std::string(reduce) +
      "'. Supported modes: sum, prod, mean, amax, amin.");
}

}  // namespace detail

// scatter_reduce: Scatter and reduce values from src into self at indices.
// Maps to Paddle's put_along_axis with appropriate reduce mode conversion.
inline at::Tensor Tensor::scatter_reduce(int64_t dim,
                                         const at::Tensor& index,
                                         const at::Tensor& src,
                                         c10::string_view reduce,
                                         bool include_self) const {
  std::string paddle_reduce = detail::_pd_scatter_reduce_str(reduce);
  int normalized_dim = detail::_pd_normalize_dim(dim, this->dim());
  detail::_pd_check_scatter_reduce_shape(*this, index, src, normalized_dim);
  detail::_pd_check_index(index);
  return Tensor(paddle::experimental::put_along_axis(tensor_,
                                                     index._PD_GetInner(),
                                                     src._PD_GetInner(),
                                                     normalized_dim,
                                                     paddle_reduce,
                                                     include_self));
}

// scatter_reduce_: In-place version.
inline at::Tensor& Tensor::scatter_reduce_(int64_t dim,
                                           const at::Tensor& index,
                                           const at::Tensor& src,
                                           c10::string_view reduce,
                                           bool include_self) const {
  auto& self = const_cast<at::Tensor&>(*this);
  std::string paddle_reduce = detail::_pd_scatter_reduce_str(reduce);
  int normalized_dim = detail::_pd_normalize_dim(dim, this->dim());
  detail::_pd_check_scatter_reduce_shape(*this, index, src, normalized_dim);
  detail::_pd_check_index(index);
  paddle::experimental::put_along_axis_(self.tensor_,
                                        index._PD_GetInner(),
                                        src._PD_GetInner(),
                                        normalized_dim,
                                        paddle_reduce,
                                        include_self);
  return self;
}

}  // namespace at
