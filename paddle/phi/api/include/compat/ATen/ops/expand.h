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
#include <ATen/ops/as_strided.h>

#include <vector>

#include "paddle/common/flags.h"
#include "paddle/phi/api/include/api.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace at {
namespace detail {

// Helper: compute PyTorch-style expand strides.
// Right-aligns tensor dimensions with target shape.  For broadcast
// dimensions (original size == 1, target size != 1) the stride is 0.
// For missing dimensions (input_rank < target_rank) the conceptual
// stride is the product of subsequent result sizes and strides.
inline std::vector<int64_t> compute_expand_strides(
    at::IntArrayRef tensor_sizes,
    at::IntArrayRef tensor_strides,
    const std::vector<int64_t>& target_sizes) {
  int64_t ndim = static_cast<int64_t>(target_sizes.size());
  int64_t tensor_dim = static_cast<int64_t>(tensor_sizes.size());

  if (tensor_dim == 0) {
    // scalar: all strides are 0
    return std::vector<int64_t>(ndim, 0);
  }

  std::vector<int64_t> expanded_strides(ndim);
  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dim = tensor_dim - 1 - offset;
    int64_t size = (dim >= 0) ? tensor_sizes[dim] : 1;
    int64_t stride = (dim >= 0) ? tensor_strides[dim]
                                : target_sizes[i + 1] * expanded_strides[i + 1];
    if (size != target_sizes[i]) {
      stride = 0;
    }
    expanded_strides[i] = stride;
  }
  return expanded_strides;
}

}  // namespace detail

// expand - expands tensor to new size
// PyTorch's expand works by right-aligning dimensions and broadcasting
// dimensions with size 1 to the target size.
// The result is a VIEW with stride=0 for broadcast dimensions when stride
// kernels are enabled.  Otherwise materialize the broadcast because Paddle's
// non-stride-kernel path cannot consume stride-0 tensors reliably.
inline Tensor expand(const Tensor& self,
                     at::IntArrayRef size,
                     bool implicit = false) {
  // implicit parameter is used by PyTorch's vmap for internal optimization.
  // It doesn't affect the actual expand operation, so we can safely ignore it.

  // Target sizes - convert to vector so we can mutate -1 entries
  std::vector<int64_t> target_size_vec(size.begin(), size.end());
  auto target_rank = target_size_vec.size();
  auto input_sizes = self.sizes();
  auto input_strides = self.strides();
  auto input_rank = static_cast<size_t>(input_sizes.size());

  if (input_rank > target_rank) {
    PD_THROW("expand(): the number of sizes provided (",
             target_rank,
             ") must be greater or equal to the number of dimensions in the "
             "tensor (",
             input_rank,
             ").");
  }

  // Validate that expansion is compatible (same logic as PyTorch)
  for (size_t i = 0; i < target_rank; ++i) {
    int64_t offset = target_rank - 1 - i;
    int64_t dim = static_cast<int64_t>(input_rank) - 1 - offset;
    int64_t in_size = (dim >= 0) ? input_sizes[dim] : 1;
    int64_t target_size = target_size_vec[i];

    if (target_size == -1) {
      if (dim < 0) {
        PD_THROW(
            "expand(): the expanded size of the tensor (-1) isn't allowed "
            "in a leading, non-existing dimension ",
            i,
            ".");
      }
      target_size_vec[i] = in_size;
      continue;
    }

    if (target_size < 0) {
      PD_THROW("expand(): the expanded size of the tensor (",
               target_size,
               ") isn't allowed in a target dimension ",
               i,
               ".");
    }

    if (in_size != target_size && in_size != 1) {
      PD_THROW("expand(): the expanded size of the tensor (",
               target_size,
               ") must match the existing size (",
               in_size,
               ") at non-singleton dimension ",
               i,
               ".");
    }
  }

  if (!FLAGS_use_stride_kernel) {
    auto input = self._PD_GetInner();
    if (input_rank < target_rank) {
      std::vector<int64_t> reshape_size(target_rank - input_rank, 1);
      reshape_size.insert(
          reshape_size.end(), input_sizes.begin(), input_sizes.end());
      input = paddle::experimental::reshape(
          input, paddle::experimental::IntArray(reshape_size));
    }
    return Tensor(paddle::experimental::expand(
        input, paddle::experimental::IntArray(target_size_vec)));
  }

  // Compute PyTorch-style strides and create a view via as_strided
  auto expected_strides = detail::compute_expand_strides(
      input_sizes, input_strides, target_size_vec);
  return self.as_strided(at::IntArrayRef(target_size_vec),
                         at::IntArrayRef(expected_strides));
}

}  // namespace at

namespace at {

// Member function: Tensor::expand
inline Tensor Tensor::expand(at::IntArrayRef size, bool implicit) const {
  return at::expand(*this, size, implicit);
}

}  // namespace at
