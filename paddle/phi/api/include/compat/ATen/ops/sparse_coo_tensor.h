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
  // When size is not provided, Paddle will infer it from indices and values
  return paddle::experimental::sparse::sparse_coo_tensor(
      values._PD_GetInner(), indices._PD_GetInner(), {});
}

}  // namespace at

namespace torch {
using at::sparse_coo_tensor;
}  // namespace torch
