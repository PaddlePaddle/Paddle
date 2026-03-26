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

inline std::vector<int64_t> infer_sparse_coo_size(const at::Tensor& indices) {
  auto host_indices = indices.cpu().to(at::kLong);
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

  if (options.pinned_memory()) {
    phi::Place base_place = options._PD_GetPlace();
    phi::Place pinned_place = compat::_PD_GetCreatePinnedPlace(base_place);
    idx = idx.copy_to(pinned_place, /*blocking=*/true);
    vals = vals.copy_to(pinned_place, /*blocking=*/true);
  }

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

  const int64_t sparse_dims = indices.size(0);
  const int64_t nnz = indices.size(1);
  std::vector<int64_t> inferred_size;
  inferred_size.reserve(sparse_dims + std::max<int64_t>(static_cast<int64_t>(0),
                                                        values.dim() - 1));

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
