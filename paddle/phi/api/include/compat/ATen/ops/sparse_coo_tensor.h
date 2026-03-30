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
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/common/place.h"

namespace at {

inline paddle::Tensor copy_dense_tensor_for_sparse_coo_if_needed(
    const paddle::Tensor& tensor, const phi::Place& place) {
  if (tensor.place() == place) {
    return tensor;
  }
  return tensor.copy_to(place, /*blocking=*/true);
}

inline void apply_sparse_coo_creation_options(
    paddle::Tensor* indices,
    paddle::Tensor* values,
    const at::TensorOptions& options) {
  if (options.pinned_memory()) {
    if (options.has_device() && !options.device().is_cpu()) {
      PD_THROW(
          "pin_memory=true requires device to be CPU, but got non-CPU device");
    }
    phi::Place pinned_place =
        compat::_PD_GetCreatePinnedPlace(options._PD_GetPlace());
    *indices =
        copy_dense_tensor_for_sparse_coo_if_needed(*indices, pinned_place);
    *values = copy_dense_tensor_for_sparse_coo_if_needed(*values, pinned_place);
    return;
  }

  if (options.has_device()) {
    const phi::Place target_place = options.device()._PD_GetInner();
    *indices =
        copy_dense_tensor_for_sparse_coo_if_needed(*indices, target_place);
    *values = copy_dense_tensor_for_sparse_coo_if_needed(*values, target_place);
  }
}

inline std::vector<int64_t> infer_sparse_coo_size(const at::Tensor& indices) {
  auto host_indices = indices.cpu().to(at::kLong).contiguous();
  int64_t sparse_dim = host_indices.dim() > 0 ? host_indices.size(0) : 0;
  int64_t nnz = host_indices.dim() > 1 ? host_indices.size(1) : 0;

  std::vector<int64_t> inferred_size(static_cast<size_t>(sparse_dim), 0);
  const int64_t* data = host_indices.const_data_ptr<int64_t>();
  for (int64_t dim = 0; dim < sparse_dim; ++dim) {
    for (int64_t i = 0; i < nnz; ++i) {
      inferred_size[static_cast<size_t>(dim)] = std::max(
          inferred_size[static_cast<size_t>(dim)], data[dim * nnz + i] + 1);
    }
  }
  return inferred_size;
}

inline at::Tensor sparse_coo_tensor(const at::Tensor& indices,
                                    const at::Tensor& values,
                                    at::IntArrayRef size,
                                    at::TensorOptions options = {}) {
  paddle::Tensor idx = indices._PD_GetInner();
  paddle::Tensor vals = values._PD_GetInner();

  if (options.dtype_opt().has_value() &&
      options.dtype_opt().value() != values.scalar_type()) {
    vals = paddle::experimental::cast(
        vals,
        compat::_PD_AtenScalarTypeToPhiDataType(options.dtype_opt().value()));
  }
  apply_sparse_coo_creation_options(&idx, &vals, options);

  // PyTorch: sparse_coo_tensor(indices, values, size)
  // Paddle:  sparse_coo_tensor(values, indices, shape)
  return paddle::experimental::sparse::sparse_coo_tensor(
      vals, idx, std::vector<int64_t>(size.begin(), size.end()));
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
  auto options =
      at::TensorOptions().dtype(dtype).device(device).pinned_memory(pin_memory);
  return sparse_coo_tensor(indices, values, size, options);
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

  std::vector<int64_t> inferred_size = infer_sparse_coo_size(indices);
  for (int64_t d = 1; d < values.dim(); ++d) {
    inferred_size.push_back(values.size(d));
  }

  return sparse_coo_tensor(
      indices, values, at::IntArrayRef(inferred_size), options);
}

}  // namespace at
