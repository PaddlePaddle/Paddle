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
#include <c10/core/TensorOptions.h>
#include <algorithm>
#include <optional>

#include "paddle/phi/api/include/sparse_api.h"

namespace at {

inline at::Tensor sparse_coo_tensor(const at::Tensor& indices,
                                    const at::Tensor& values,
                                    at::IntArrayRef size,
                                    at::TensorOptions options = {}) {
  // PyTorch: sparse_coo_tensor(indices, values, size)
  // Paddle:  sparse_coo_tensor(values, indices, shape)
  return paddle::experimental::sparse::sparse_coo_tensor(
      values._PD_GetInner(),
      indices._PD_GetInner(),
      std::vector<int64_t>(size.begin(), size.end()));
}

inline at::Tensor sparse_coo_tensor(const at::Tensor& indices,
                                    const at::Tensor& values,
                                    at::IntArrayRef size,
                                    ::std::optional<at::ScalarType> dtype,
                                    ::std::optional<at::Layout> layout,
                                    ::std::optional<at::Device> device,
                                    ::std::optional<bool> pin_memory) {
  PD_CHECK(!layout.has_value() || layout.value() == c10::kSparse,
           "`layout` must be Sparse for sparse_coo_tensor.");
  PD_CHECK(!(pin_memory.has_value() && pin_memory.value() != false),
           "`pin_memory` other than False is not supported now.");

  // Note: dtype and device are used for validation/casting if needed
  // Currently, we use the values tensor's dtype and device
  return paddle::experimental::sparse::sparse_coo_tensor(
      values._PD_GetInner(),
      indices._PD_GetInner(),
      std::vector<int64_t>(size.begin(), size.end()));
}

inline at::Tensor sparse_coo_tensor(const at::Tensor& indices,
                                    const at::Tensor& values,
                                    at::TensorOptions options = {}) {
    // PyTorch 语义：未提供 size 时根据 indices/values 推断完整 shape。
    // size = [max(indices[d]) + 1 for d in sparse_dims] + values.shape[1:]
    PD_CHECK(indices.dim() == 2,
                     "`indices` for sparse_coo_tensor must be a 2-D tensor, but got ",
                     indices.dim(),
                     "-D tensor.");

    PD_CHECK(indices.scalar_type() == at::kLong,
                     "`indices` for sparse_coo_tensor must have dtype int64.");

    const int64_t sparse_dims = indices.size(0);
    const int64_t nnz = indices.size(1);
    std::vector<int64_t> inferred_size;
    inferred_size.reserve(
            sparse_dims + std::max<int64_t>(int64_t(0), values.dim() - 1));

    PD_CHECK(indices.is_cpu(),
                     "`indices` must be on CPU when inferring sparse_coo_tensor size.");
    auto idx_tensor = indices.contiguous()._PD_GetInner();
    const int64_t* idx_data = idx_tensor.data<int64_t>();

    for (int64_t d = 0; d < sparse_dims; ++d) {
        int64_t dim_size = 0;
        for (int64_t i = 0; i < nnz; ++i) {
            dim_size = std::max(dim_size, idx_data[d * nnz + i] + 1);
        }
        inferred_size.push_back(dim_size);
    }

    for (int64_t d = 1; d < values.dim(); ++d) {
        inferred_size.push_back(values.size(d));
    }

  return paddle::experimental::sparse::sparse_coo_tensor(
            values._PD_GetInner(), indices._PD_GetInner(), inferred_size);
}

}  // namespace at

namespace torch {
using at::sparse_coo_tensor;
}  // namespace torch
