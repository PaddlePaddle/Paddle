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
#include <utils/pinned_place.h>
#include <optional>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace at {

inline at::Tensor sparse_csr_tensor(const at::Tensor& crow_indices,
                                    const at::Tensor& col_indices,
                                    const at::Tensor& values,
                                    at::IntArrayRef size,
                                    at::TensorOptions options = {}) {
  paddle::Tensor crows = crow_indices._PD_GetInner();
  paddle::Tensor cols = col_indices._PD_GetInner();
  paddle::Tensor vals = values._PD_GetInner();

  if (options.dtype_opt().has_value() &&
      options.dtype_opt().value() != values.scalar_type()) {
    vals = paddle::experimental::cast(
        vals,
        compat::_PD_AtenScalarTypeToPhiDataType(options.dtype_opt().value()));
  }

  if (options.pinned_memory()) {
    phi::Place base_place = options._PD_GetPlace();
    phi::Place pinned_place = compat::_PD_GetCreatePinnedPlace(base_place);
    crows = crows.copy_to(pinned_place, /*blocking=*/true);
    cols = cols.copy_to(pinned_place, /*blocking=*/true);
    vals = vals.copy_to(pinned_place, /*blocking=*/true);
  }

  // Get the underlying DenseTensors
  auto* dense_crows = dynamic_cast<phi::DenseTensor*>(crows.impl().get());
  auto* dense_cols = dynamic_cast<phi::DenseTensor*>(cols.impl().get());
  auto* dense_values = dynamic_cast<phi::DenseTensor*>(vals.impl().get());

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
  auto options =
      at::TensorOptions().dtype(dtype).device(device).pinned_memory(pin_memory);
  return sparse_csr_tensor(crow_indices, col_indices, values, size, options);
}

inline at::Tensor sparse_csr_tensor(const at::Tensor& crow_indices,
                                    const at::Tensor& col_indices,
                                    const at::Tensor& values,
                                    at::TensorOptions options = {}) {
  // Infer size from crow_indices and col_indices:
  //   nrows = crow_indices.size(0) - 1
  //   ncols = max(col_indices) + 1
  int64_t nrows = crow_indices.size(0) - 1;
  int64_t ncols = 0;

  if (col_indices.numel() > 0) {
    auto* dense_cols = dynamic_cast<phi::DenseTensor*>(
        col_indices._PD_GetInner().impl().get());
    PD_CHECK(dense_cols != nullptr,
             "col_indices must be a dense tensor for sparse_csr_tensor.");
    PD_CHECK(
        dense_cols->place().GetType() == phi::AllocationType::CPU,
        "sparse_csr_tensor without explicit size only supports CPU "
        "col_indices for automatic size inference. Please provide the size "
        "parameter explicitly for non-CPU tensors.");

    int64_t n = dense_cols->numel();
    if (dense_cols->dtype() == phi::DataType::INT64) {
      const int64_t* data = dense_cols->data<int64_t>();
      for (int64_t i = 0; i < n; ++i) {
        if (data[i] + 1 > ncols) ncols = data[i] + 1;
      }
    } else if (dense_cols->dtype() == phi::DataType::INT32) {
      const int32_t* data = dense_cols->data<int32_t>();
      for (int64_t i = 0; i < n; ++i) {
        int64_t val = static_cast<int64_t>(data[i]) + 1;
        if (val > ncols) ncols = val;
      }
    } else {
      PD_CHECK(false,
               "col_indices must have dtype int32 or int64 for automatic "
               "size inference in sparse_csr_tensor.");
    }
  }

  std::vector<int64_t> size_vec = {nrows, ncols};
  return sparse_csr_tensor(
      crow_indices, col_indices, values, at::IntArrayRef(size_vec), options);
}

}  // namespace at
