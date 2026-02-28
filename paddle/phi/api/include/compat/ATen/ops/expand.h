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

#include "paddle/phi/api/include/api.h"

namespace at {

// expand - expands tensor to new size
// PyTorch's expand works by right-aligning dimensions and broadcasting
// dimensions with size 1 to the target size
// Unlike Paddle's expand_v2, PyTorch allows non-singleton dimensions to be
// preserved when they match the corresponding target dimension
inline Tensor expand(const Tensor& self,
                     at::IntArrayRef size,
                     bool implicit = false) {
  // implicit parameter is used by PyTorch's vmap for internal optimization.
  // It doesn't affect the actual expand operation, so we can safely ignore it.

  paddle::Tensor pd_tensor = self._PD_GetInner();

  // Target sizes - convert to vector
  std::vector<int64_t> target_size_vec(size.begin(), size.end());
  auto target_rank = target_size_vec.size();
  auto input_dims = pd_tensor.dims();
  auto input_rank = input_dims.size();

  auto tile_and_slice_to_target =
      [&](const paddle::Tensor& input,
          const std::vector<int64_t>& input_shape,
          const std::vector<int64_t>& target_shape) -> Tensor {
    size_t rank = target_shape.size();
    std::vector<int64_t> repeat_times(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
      auto in_size = input_shape[i];
      auto target_size = target_shape[i];

      if (in_size == 0 || target_size == 0) {
        repeat_times[i] = 0;
      } else if (target_size <= in_size) {
        repeat_times[i] = 1;
      } else {
        repeat_times[i] = (target_size + in_size - 1) / in_size;
      }
    }

    paddle::Tensor tiled =
        paddle::experimental::tile(input, phi::IntArray(repeat_times));

    std::vector<int64_t> axes(rank);
    std::vector<int64_t> starts(rank, 0);
    std::vector<int64_t> ends(rank);
    std::vector<int64_t> strides(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
      axes[i] = static_cast<int64_t>(i);
      ends[i] = target_shape[i];
    }

    paddle::Tensor sliced =
        paddle::experimental::slice(tiled, axes, starts, ends, strides, {});
    return Tensor(sliced);
  };

  // PyTorch's expand uses right-alignment semantics:
  // - For 1D tensor expand to 2D: {3}.expand({3,4}) treats input as {3,1},
  // expands to {3,4}
  // - Non-singleton dimensions are preserved, singleton dimensions (1) can
  // expand
  //
  // For example:
  //   {3}.expand({3, 4}) -> input {3} becomes {3, 1} implicitly
  //   then expand: dim 0: 3 stays 3, dim 1: 1 -> 4 -> result {3, 4}

  if (input_rank < target_rank) {
    // Add trailing 1s to right-align with target shape (PyTorch behavior)
    // Input {3}, target {3, 4} -> reshape to {3, 1}
    std::vector<int64_t> reshape_vec(input_rank, 1);
    for (size_t i = 0; i < input_rank; ++i) {
      reshape_vec[i] = input_dims[i];
    }
    // Add trailing 1s
    while (reshape_vec.size() < target_rank) {
      reshape_vec.push_back(1);
    }

    // Check if Paddle's expand can handle this right-aligned shape
    // Paddle allows: input[i] == 1 (can expand), or input[i] == target[i]
    // (match)
    bool can_use_paddle_expand = true;
    for (size_t i = 0; i < target_rank; ++i) {
      bool dim_can_expand = (reshape_vec[i] == 1);
      bool dim_is_matching = (reshape_vec[i] == target_size_vec[i]);
      if (!dim_can_expand && !dim_is_matching) {
        can_use_paddle_expand = false;
        break;
      }
    }

    if (can_use_paddle_expand) {
      // Reshape to right-aligned shape, then expand
      paddle::Tensor reshaped =
          paddle::experimental::reshape(pd_tensor, phi::IntArray(reshape_vec));
      paddle::Tensor result = paddle::experimental::expand(
          reshaped, phi::IntArray(target_size_vec));
      return Tensor(result);
    }

    // If Paddle's expand can't handle it, use tile + slice as fallback
    paddle::Tensor reshaped =
        paddle::experimental::reshape(pd_tensor, phi::IntArray(reshape_vec));
    return tile_and_slice_to_target(reshaped, reshape_vec, target_size_vec);
  } else if (input_rank == target_rank) {
    // Same rank - check if we can use expand directly or need tile
    bool can_use_paddle_expand = true;
    for (size_t i = 0; i < target_rank; ++i) {
      auto in_size = input_dims[i];
      auto target_size = target_size_vec[i];
      if (in_size != 1 && in_size != target_size) {
        can_use_paddle_expand = false;
        break;
      }
    }

    if (can_use_paddle_expand) {
      paddle::Tensor result = paddle::experimental::expand(
          pd_tensor, phi::IntArray(target_size_vec));
      return Tensor(result);
    }

    // Need tile + slice fallback
    std::vector<int64_t> input_shape(target_rank);
    for (size_t i = 0; i < target_rank; ++i) {
      input_shape[i] = input_dims[i];
    }
    return tile_and_slice_to_target(pd_tensor, input_shape, target_size_vec);
  } else {
    // Input has more dimensions.
    // Keep the trailing target_rank dimensions and slice leading dimensions to
    // 1 before reshape, so total element count remains valid.
    paddle::Tensor squeezed = pd_tensor;
    size_t leading_dims = input_rank - target_rank;
    for (size_t i = 0; i < leading_dims; ++i) {
      squeezed = paddle::experimental::slice(
          squeezed, {static_cast<int64_t>(i)}, {0}, {1}, {1}, {});
    }

    std::vector<int64_t> new_shape(target_rank);
    for (size_t i = 0; i < target_rank; ++i) {
      new_shape[i] = input_dims[i + (input_rank - target_rank)];
    }

    // Reshape to target rank, then reuse the same expand implementation.
    paddle::Tensor reshaped =
        paddle::experimental::reshape(squeezed, phi::IntArray(new_shape));

    return expand(Tensor(reshaped), size, implicit);
  }
}

// expand_as - expands to same size as another tensor
inline Tensor expand_as(const Tensor& self, const Tensor& other) {
  return expand(self, other.sizes());
}

}  // namespace at

namespace at {

// Member function: Tensor::expand
inline Tensor Tensor::expand(at::IntArrayRef size, bool implicit) const {
  return at::expand(*this, size, implicit);
}

// Member function: Tensor::expand_as
inline Tensor Tensor::expand_as(const Tensor& other) const {
  return at::expand_as(*this, other);
}

}  // namespace at
