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
#include <c10/core/TensorOptions.h>
#include <optional>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace at {

inline at::Tensor sparse_csr_tensor(const at::Tensor& crow_indices,
                                    const at::Tensor& col_indices,
                                    const at::Tensor& values,
                                    at::IntArrayRef size,
                                    at::TensorOptions options = {}) {
  // Get the underlying DenseTensors
  auto* dense_crows =
      dynamic_cast<phi::DenseTensor*>(crow_indices._PD_GetInner().impl().get());
  auto* dense_cols =
      dynamic_cast<phi::DenseTensor*>(col_indices._PD_GetInner().impl().get());
  auto* dense_values =
      dynamic_cast<phi::DenseTensor*>(values._PD_GetInner().impl().get());

  PD_CHECK(dense_crows != nullptr,
           "crow_indices must be a dense tensor for sparse_csr_tensor.");
  PD_CHECK(dense_cols != nullptr,
           "col_indices must be a dense tensor for sparse_csr_tensor.");
  PD_CHECK(dense_values != nullptr,
           "values must be a dense tensor for sparse_csr_tensor.");

  // Create the SparseCsrTensor
  std::shared_ptr<phi::SparseCsrTensor> csr_tensor =
      std::make_shared<phi::SparseCsrTensor>(
          *dense_crows,
          *dense_cols,
          *dense_values,
          common::make_ddim(std::vector<int64_t>(size.begin(), size.end())));

  // Wrap in a Paddle Tensor
  paddle::Tensor result;
  result.set_impl(csr_tensor);
  return result;
}

inline at::Tensor sparse_csr_tensor(const at::Tensor& crow_indices,
                                    const at::Tensor& col_indices,
                                    const at::Tensor& values,
                                    at::IntArrayRef size,
                                    ::std::optional<at::ScalarType> dtype,
                                    ::std::optional<at::Layout> layout,
                                    ::std::optional<at::Device> device,
                                    ::std::optional<bool> pin_memory) {
  PD_CHECK(!layout.has_value() || layout.value() == c10::kSparseCsr,
           "`layout` must be SparseCsr for sparse_csr_tensor.");
  PD_CHECK(!(pin_memory.has_value() && pin_memory.value() != false),
           "`pin_memory` other than False is not supported now.");

  return sparse_csr_tensor(crow_indices, col_indices, values, size);
}

inline at::Tensor sparse_csr_tensor(const at::Tensor& crow_indices,
                                    const at::Tensor& col_indices,
                                    const at::Tensor& values,
                                    at::TensorOptions options = {}) {
  // Infer size from crow_indices and col_indices
  // nrows = crow_indices.size(0) - 1
  // ncols = max(col_indices) + 1 (approximated by col_indices for now)
  int64_t nrows = crow_indices.size(0) - 1;

  // For ncols, we need to find the maximum value in col_indices + 1
  // For simplicity, we require explicit size in this case
  PD_CHECK(false,
           "sparse_csr_tensor without explicit size is not supported. "
           "Please provide the size parameter.");
  return at::Tensor();
}

}  // namespace at

namespace torch {
using at::sparse_csr_tensor;
}  // namespace torch
