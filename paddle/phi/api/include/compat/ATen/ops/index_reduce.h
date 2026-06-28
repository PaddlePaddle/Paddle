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
#include <ATen/ops/item.h>
#include <c10/util/string_view.h>

#include "paddle/phi/api/include/api.h"

namespace at {

namespace detail {

// Helper: convert PyTorch reduce name to Paddle reduce name
inline std::string convert_reduce_name(c10::string_view reduce) {
  if (reduce == "prod") {
    return "multiply";
  }
  PD_CHECK(reduce == "mean" || reduce == "amax" || reduce == "amin",
           "index_reduce(): reduce must be one of 'prod', 'mean', "
           "'amax' or 'amin', but got '",
           reduce,
           "'.");
  return std::string(reduce);
}

// Helper: normalize dim and validate inputs for index_reduce.
// Returns the positive (normalized) dim value.
inline int64_t normalize_index_reduce_dim(const at::Tensor& self,
                                          const at::Tensor& index,
                                          const at::Tensor& source,
                                          int64_t dim) {
  const int64_t ndim = self.dim();
  const int64_t original_dim = dim;
  if (dim < 0) {
    dim += ndim;
  }
  PD_CHECK(dim >= 0 && dim < ndim,
           "index_reduce(): dim out of range (expected to be in range of [",
           -ndim,
           ", ",
           ndim - 1,
           "], but got ",
           original_dim,
           ")");
  PD_CHECK(source.dim() == ndim,
           "index_reduce(): source rank must match self rank");
  PD_CHECK(self.scalar_type() == source.scalar_type(),
           "index_reduce(): self and source must have the same scalar type");
  PD_CHECK(index.dim() == 1, "index_reduce(): index must be a 1-D tensor");
  PD_CHECK(index.scalar_type() == at::kLong || index.scalar_type() == at::kInt,
           "index_reduce(): Expected dtype int32/int64 for index but got ",
           index.scalar_type());
  PD_CHECK(index.size(0) == source.size(dim),
           "index_reduce(): index length must match source.size(dim)");
  for (int64_t i = 0; i < ndim; ++i) {
    if (i != dim) {
      PD_CHECK(
          source.size(i) == self.size(i),
          "index_reduce(): source dimensions other than dim must match self");
    }
  }
  return dim;
}

// Helper: PyTorch index_reduce rejects index values outside [0,
// self.size(dim)).
inline void check_index_reduce_index_bounds(const at::Tensor& index,
                                            int64_t upper_bound) {
  if (index.numel() == 0) {
    return;
  }

  auto min_tensor = at::Tensor(paddle::experimental::min(index._PD_GetInner()));
  auto max_tensor = at::Tensor(paddle::experimental::max(index._PD_GetInner()));
  int64_t min_value = 0;
  int64_t max_value = 0;
  if (index.scalar_type() == at::kLong) {
    min_value = min_tensor.item<int64_t>();
    max_value = max_tensor.item<int64_t>();
  } else {
    min_value = static_cast<int64_t>(min_tensor.item<int>());
    max_value = static_cast<int64_t>(max_tensor.item<int>());
  }

  PD_CHECK(min_value >= 0 && max_value < upper_bound,
           "index_reduce(): index values must be in range [0, ",
           upper_bound,
           "), but found range [",
           min_value,
           ", ",
           max_value,
           "]");
}

// Helper: expand 1D index to match source shape for put_along_axis.
// PyTorch index_reduce uses a 1D index where each entry maps a slice of source
// to a slice of self along the operation dimension. Paddle put_along_axis
// requires index and source to have matching shape (or broadcastable).
// This helper broadcasts the 1D index across all non-operation dimensions.
inline paddle::Tensor expand_index_for_put_along_axis(
    const paddle::Tensor& index, const paddle::Tensor& source, int64_t dim) {
  if (index.dims().size() == 1 && source.dims().size() > 1) {
    // Reshape index: [N] -> [1, ..., N, ..., 1] where N is at dim position
    std::vector<int64_t> reshape_shape(source.dims().size(), 1);
    reshape_shape[dim] = index.dims()[0];
    auto reshaped = paddle::experimental::reshape(
        index, paddle::experimental::IntArray(reshape_shape));

    // Expand to source shape
    std::vector<int64_t> expand_shape;
    for (int i = 0; i < source.dims().size(); ++i) {
      expand_shape.push_back(source.dims()[i]);
    }
    return paddle::experimental::expand(
        reshaped, paddle::experimental::IntArray(expand_shape));
  }
  return index;
}

}  // namespace detail

// index_reduce: out-of-place version
inline at::Tensor index_reduce(const at::Tensor& self,
                               int64_t dim,
                               const at::Tensor& index,
                               const at::Tensor& source,
                               c10::string_view reduce,
                               bool include_self = true) {
  auto pd_reduce = detail::convert_reduce_name(reduce);
  auto pd_dim = detail::normalize_index_reduce_dim(self, index, source, dim);
  detail::check_index_reduce_index_bounds(index, self.size(pd_dim));
  auto pd_index = detail::expand_index_for_put_along_axis(
      index._PD_GetInner(), source._PD_GetInner(), pd_dim);
  auto result = paddle::experimental::put_along_axis(self._PD_GetInner(),
                                                     pd_index,
                                                     source._PD_GetInner(),
                                                     static_cast<int>(pd_dim),
                                                     pd_reduce,
                                                     include_self);
  return at::Tensor(result);
}

// index_reduce_out: out variant with pre-allocated output tensor
inline at::Tensor& index_reduce_out(
    at::Tensor& out,  // NOLINT(runtime/references)
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    c10::string_view reduce,
    bool include_self = true) {
  auto pd_reduce = detail::convert_reduce_name(reduce);
  auto pd_dim = detail::normalize_index_reduce_dim(self, index, source, dim);
  detail::check_index_reduce_index_bounds(index, self.size(pd_dim));
  auto pd_index = detail::expand_index_for_put_along_axis(
      index._PD_GetInner(), source._PD_GetInner(), pd_dim);
  auto result = paddle::experimental::put_along_axis(self._PD_GetInner(),
                                                     pd_index,
                                                     source._PD_GetInner(),
                                                     static_cast<int>(pd_dim),
                                                     pd_reduce,
                                                     include_self);
  paddle::experimental::assign_out_(result, out._PD_GetInner());
  return out;
}

// index_reduce_outf: out variant with output as last parameter
inline at::Tensor& index_reduce_outf(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    c10::string_view reduce,
    bool include_self,
    at::Tensor& out) {  // NOLINT(runtime/references)
  return index_reduce_out(out, self, dim, index, source, reduce, include_self);
}

namespace detail {

// index_reduce_: in-place version
inline at::Tensor& index_reduce_(
    at::Tensor& self,  // NOLINT(runtime/references)
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    c10::string_view reduce,
    bool include_self = true) {
  auto pd_reduce = convert_reduce_name(reduce);
  auto pd_dim = normalize_index_reduce_dim(self, index, source, dim);
  check_index_reduce_index_bounds(index, self.size(pd_dim));
  auto pd_index = expand_index_for_put_along_axis(
      index._PD_GetInner(), source._PD_GetInner(), pd_dim);
  paddle::experimental::put_along_axis_(self._PD_GetInner(),
                                        pd_index,
                                        source._PD_GetInner(),
                                        static_cast<int>(pd_dim),
                                        pd_reduce,
                                        include_self);
  return self;
}

}  // namespace detail

}  // namespace at

namespace at {

// Tensor member functions
inline at::Tensor Tensor::index_reduce(int64_t dim,
                                       const at::Tensor& index,
                                       const at::Tensor& source,
                                       c10::string_view reduce,
                                       bool include_self) const {
  return at::index_reduce(*this, dim, index, source, reduce, include_self);
}

inline at::Tensor& Tensor::index_reduce_(int64_t dim,
                                         const at::Tensor& index,
                                         const at::Tensor& source,
                                         c10::string_view reduce,
                                         bool include_self) const {
  return at::detail::index_reduce_(
      const_cast<at::Tensor&>(*this), dim, index, source, reduce, include_self);
}

}  // namespace at
