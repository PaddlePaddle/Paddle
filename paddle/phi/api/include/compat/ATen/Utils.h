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

#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>

#include <algorithm>

#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/include/tensor.h"

namespace at {
namespace detail {

// Helper function to convert dense tensor to sparse based on layout
inline at::Tensor _PD_ConvertToSparseIfNeeded(
    const paddle::Tensor& dense_tensor, c10::Layout layout) {
  switch (layout) {
    case c10::kStrided:
      return dense_tensor;
    case c10::kSparse:
      // Convert to sparse COO format, sparse_dim = number of dimensions
      return paddle::experimental::sparse::to_sparse_coo(
          dense_tensor, dense_tensor.dims().size());
    case c10::kSparseCsr:
      return paddle::experimental::sparse::to_sparse_csr(dense_tensor);
    default:
      PD_CHECK(false,
               "Unsupported layout for sparse tensor creation. "
               "Supported layouts: Strided, Sparse (COO), SparseCsr.");
  }
}

}  // namespace detail
}  // namespace at
