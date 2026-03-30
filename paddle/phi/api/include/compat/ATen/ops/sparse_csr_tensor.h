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

#include <algorithm>
#include <optional>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace at {

inline paddle::Tensor copy_dense_tensor_for_sparse_csr_if_needed(
    const paddle::Tensor& tensor, const phi::Place& place) {
  if (tensor.place() == place) {
    return tensor;
  }
  return tensor.copy_to(place, /*blocking=*/true);
}

inline void apply_sparse_csr_creation_options(
    paddle::Tensor* crow_indices,
    paddle::Tensor* col_indices,
    paddle::Tensor* values,
    const at::TensorOptions& options) {
  if (options.pinned_memory()) {
    if (options.has_device() && !options.device().is_cpu()) {
      PD_THROW(
          "pin_memory=true requires device to be CPU, but got non-CPU device");
    }
    phi::Place pinned_place =
        compat::_PD_GetCreatePinnedPlace(options._PD_GetPlace());
    *crow_indices =
        copy_dense_tensor_for_sparse_csr_if_needed(*crow_indices, pinned_place);
    *col_indices =
        copy_dense_tensor_for_sparse_csr_if_needed(*col_indices, pinned_place);
    *values = copy_dense_tensor_for_sparse_csr_if_needed(*values, pinned_place);
    return;
  }

  if (options.has_device()) {
    const phi::Place target_place = options.device()._PD_GetInner();
    *crow_indices =
        copy_dense_tensor_for_sparse_csr_if_needed(*crow_indices, target_place);
    *col_indices =
        copy_dense_tensor_for_sparse_csr_if_needed(*col_indices, target_place);
    *values = copy_dense_tensor_for_sparse_csr_if_needed(*values, target_place);
  }
}

inline int64_t infer_sparse_csr_ncols(const at::Tensor& col_indices) {
  auto host_cols = col_indices.cpu().contiguous();
  int64_t ncols = 0;
  if (host_cols.scalar_type() == at::kLong) {
    const int64_t* data = host_cols.const_data_ptr<int64_t>();
    for (int64_t i = 0; i < host_cols.numel(); ++i) {
      ncols = std::max(ncols, data[i] + 1);
    }
    return ncols;
  }
  if (host_cols.scalar_type() == at::kInt) {
    const int32_t* data = host_cols.const_data_ptr<int32_t>();
    for (int64_t i = 0; i < host_cols.numel(); ++i) {
      ncols = std::max(ncols, static_cast<int64_t>(data[i]) + 1);
    }
    return ncols;
  }
  PD_CHECK(false,
           "col_indices must have dtype int32 or int64 for automatic "
           "size inference in sparse_csr_tensor.");
  return 0;
}

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
  apply_sparse_csr_creation_options(&crows, &cols, &vals, options);

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
  int64_t ncols =
      col_indices.numel() > 0 ? infer_sparse_csr_ncols(col_indices) : 0;

  std::vector<int64_t> size_vec = {nrows, ncols};
  return sparse_csr_tensor(
      crow_indices, col_indices, values, at::IntArrayRef(size_vec), options);
}

}  // namespace at
