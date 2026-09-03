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
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/item.h>
#include <ATen/ops/sum.h>
#include <limits>
#include <vector>

#include "paddle/phi/api/include/api.h"

namespace at {

// at::repeat_interleave(repeats_tensor, output_size) - standalone function
// Returns indices [0 repeated repeats[0] times, 1 repeated repeats[1] times,
// ...]
inline at::Tensor repeat_interleave(
    const at::Tensor& repeats,
    ::std::optional<int64_t> output_size = ::std::nullopt) {
  PD_CHECK(repeats.dim() == 1,
           "repeat_interleave only accept 1D vector as repeat");
  PD_CHECK(
      repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt,
      "repeats has to be Long or Int tensor");
  if (output_size.has_value()) {
    PD_CHECK(output_size.value() >= 0,
             "repeat_interleave: output_size must be non-negative");
    if (output_size.value() == 0) {
      return at::empty({0}, repeats.options());
    }
  }

  if (repeats.numel() == 0) {
    return at::empty({0}, repeats.options());
  }

  // Create indices [0, 1, ..., n-1] with same dtype as repeats
  auto indices_options = repeats.options();
  auto end_scalar = c10::Scalar(static_cast<int64_t>(repeats.numel()));
  auto indices = at::arange(end_scalar, indices_options);

  int64_t pd_output_size = output_size.has_value() ? output_size.value() : -1;
  return Tensor(paddle::experimental::repeat_interleave_with_tensor_index(
      indices._PD_GetInner(), repeats._PD_GetInner(), 0, pd_output_size));
}

}  // namespace at

namespace at {

// Helper: validate dim and compute wrapped axis
namespace detail {
inline int wrap_and_validate_dim(int64_t dim_val, int64_t ndim) {
  PD_CHECK(ndim > 0, "repeat_interleave: input tensor has no dimensions");
  PD_CHECK(dim_val >= -ndim && dim_val < ndim,
           "repeat_interleave: dim ",
           dim_val,
           " is out of range [",
           -ndim,
           ", ",
           ndim,
           ")");
  int64_t axis64 = dim_val;
  if (axis64 < 0) {
    axis64 += ndim;
  }
  PD_CHECK(axis64 <= std::numeric_limits<int>::max(),
           "repeat_interleave: dim ",
           axis64,
           " exceeds int max");
  return static_cast<int>(axis64);
}
}  // namespace detail

// Tensor::repeat_interleave with scalar repeats
// PyTorch signature: Tensor repeat_interleave(int64_t repeats, int? dim=None,
// *, SymInt? output_size=None)
inline at::Tensor Tensor::repeat_interleave(
    int64_t repeats,
    ::std::optional<int64_t> dim,
    ::std::optional<int64_t> output_size) const {
  PD_CHECK(repeats >= 0, "Repeats must be non-negative");

  auto pd_input = tensor_;
  int axis = 0;

  if (!dim.has_value()) {
    // Flatten input if dim is not specified (PyTorch behavior)
    pd_input = paddle::experimental::flatten(pd_input, 0, -1);
    axis = 0;
  } else {
    axis = detail::wrap_and_validate_dim(dim.value(), pd_input.dims().size());
  }

  if (repeats == 0) {
    // Validate output_size before early return
    if (output_size.has_value()) {
      auto input_size = pd_input.dims()[axis];
      auto expected_size = input_size * repeats;
      PD_CHECK(expected_size == output_size.value(),
               "repeat_interleave: Invalid output_size, expected ",
               expected_size,
               " but got ",
               output_size.value());
    }
    // Paddle doesn't support repeats=0, simulate with unsqueeze+expand+flatten
    auto unsqueezed = paddle::experimental::unsqueeze(
        pd_input,
        phi::IntArray(std::vector<int64_t>{static_cast<int64_t>(axis + 1)}));
    auto expand_shape = unsqueezed.dims();
    std::vector<int64_t> shape_vec;
    for (int i = 0; i < expand_shape.size(); ++i) {
      shape_vec.push_back(expand_shape[i]);
    }
    shape_vec[axis + 1] = 0;
    auto expanded =
        paddle::experimental::expand(unsqueezed, phi::IntArray(shape_vec));
    return Tensor(paddle::experimental::flatten(expanded, axis, axis + 1));
  }

  // Narrowing check: repeats must fit in int
  PD_CHECK(repeats <= std::numeric_limits<int>::max(),
           "repeat_interleave: repeats ",
           repeats,
           " exceeds int max");

  int64_t pd_output_size = output_size.has_value() ? output_size.value() : -1;

  if (output_size.has_value()) {
    // Validate output_size
    auto input_size = pd_input.dims()[axis];
    auto expected_size = input_size * repeats;
    PD_CHECK(expected_size == pd_output_size,
             "repeat_interleave: Invalid output_size, expected ",
             expected_size,
             " but got ",
             pd_output_size);
  }

  return Tensor(paddle::experimental::repeat_interleave(
      pd_input, static_cast<int>(repeats), axis, pd_output_size));
}

// Tensor::repeat_interleave with tensor repeats
// PyTorch signature: Tensor repeat_interleave(Tensor repeats, int? dim=None,
// *, SymInt? output_size=None)
inline at::Tensor Tensor::repeat_interleave(
    const at::Tensor& repeats,
    ::std::optional<int64_t> dim,
    ::std::optional<int64_t> output_size) const {
  auto pd_input = tensor_;
  int axis = 0;

  if (!dim.has_value()) {
    // Flatten input if dim is not specified (PyTorch behavior)
    pd_input = paddle::experimental::flatten(pd_input, 0, -1);
    axis = 0;
  } else {
    axis = detail::wrap_and_validate_dim(dim.value(), pd_input.dims().size());
  }

  auto pd_repeats = repeats._PD_GetInner();

  // Handle 0-dim or scalar repeats tensor (PyTorch allows this)
  if (repeats.dim() == 0 || (repeats.dim() == 1 && repeats.numel() == 1)) {
    // Reshape to {1} and expand to match input size along axis
    pd_repeats = paddle::experimental::reshape(pd_repeats, phi::IntArray({1}));
    auto input_size = pd_input.dims()[axis];
    pd_repeats = paddle::experimental::expand(
        pd_repeats, phi::IntArray(std::vector<int64_t>{input_size}));
  } else if (repeats.dim() == 1) {
    auto repeats_size = repeats.numel();
    auto input_size = pd_input.dims()[axis];
    PD_CHECK(repeats_size == input_size,
             "repeats must have the same size as input along dim");
  } else {
    PD_CHECK(false, "repeats must be 0-dim or 1-dim tensor");
  }

  int64_t pd_output_size = -1;
  if (output_size.has_value()) {
    PD_CHECK(output_size.value() >= 0,
             "repeat_interleave: output_size must be non-negative");
    int64_t expected_size = at::Tensor(pd_repeats).sum().item<int64_t>();
    PD_CHECK(output_size.value() == expected_size,
             "repeat_interleave: Invalid output_size, expected ",
             expected_size,
             " but got ",
             output_size.value());
    pd_output_size = output_size.value();
  }
  return Tensor(paddle::experimental::repeat_interleave_with_tensor_index(
      pd_input, pd_repeats, axis, pd_output_size));
}

}  // namespace at
