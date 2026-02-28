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

    // If Paddle's expand can't handle it, use tile as fallback
    // Compute repeat_times: target_size / reshape_size
    std::vector<int64_t> repeat_times(target_rank, 1);
    for (size_t i = 0; i < target_rank; ++i) {
      if (reshape_vec[i] == 0) {
        repeat_times[i] = 0;
      } else {
        repeat_times[i] = target_size_vec[i] / reshape_vec[i];
      }
    }

    paddle::Tensor reshaped =
        paddle::experimental::reshape(pd_tensor, phi::IntArray(reshape_vec));
    paddle::Tensor result =
        paddle::experimental::tile(reshaped, phi::IntArray(repeat_times));
    return Tensor(result);
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

    // Need tile fallback
    std::vector<int64_t> repeat_times(target_rank, 1);
    for (size_t i = 0; i < target_rank; ++i) {
      if (input_dims[i] == 0) {
        repeat_times[i] = 0;
      } else {
        repeat_times[i] = target_size_vec[i] / input_dims[i];
      }
    }

    paddle::Tensor result =
        paddle::experimental::tile(pd_tensor, phi::IntArray(repeat_times));
    return Tensor(result);
  } else {
    // Input has more dimensions - need to reshape first
    // Example: {2,3,4}.expand({3,4}) -> reshape to {3,4}, then expand
    // We take the last target_rank dimensions from input
    std::vector<int64_t> new_shape(target_rank);
    for (size_t i = 0; i < target_rank; ++i) {
      new_shape[i] = input_dims[i + (input_rank - target_rank)];
    }

    // First reshape to squeeze out extra dimensions
    paddle::Tensor reshaped =
        paddle::experimental::reshape(pd_tensor, phi::IntArray(new_shape));

    // Then expand to target size
    paddle::Tensor result =
        paddle::experimental::expand(reshaped, phi::IntArray(target_size_vec));
    return Tensor(result);
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
