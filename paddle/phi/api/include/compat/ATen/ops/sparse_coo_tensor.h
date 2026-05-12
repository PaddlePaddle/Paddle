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

  // PyTorch ignores dtype mismatch between values and TensorOptions in
  // sparse_coo_tensor; the resulting sparse tensor uses values' original dtype.
  // Do not cast or throw here.

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
  return sparse_coo_tensor(
      indices, values, infer_sparse_coo_size(indices), options);
}

}  // namespace at
