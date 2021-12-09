/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/pten/core/dense_tensor.h"

namespace pten {

template <typename T>
inline bool is_zero(const T* data, const size_t n) {
  const T zero = static_cast<T>(0);
  for (size_t i = 0; i < n; i++) {
    if (data[i] != zero) {
      return false;
    }
  }
  return true;
}

template <typename T>
inline int64_t get_non_zero_num(const DenseTensor& dense,
                                const int64_t sparse_dim) {
  const auto& dims = dense.dims();
  PADDLE_ENFORCE_GE(
      dims.size(),
      sparse_dim,
      paddle::platform::errors::InvalidArgument(
          "sparse_dim(%d) should be less than or equal to dense.dim(%d)",
          sparse_dim,
          dims.size()));

  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int rows = dims_2d[0];
  const int cols = dims_2d[1];

  const T* data = dense.data<T>();
  int64_t non_zero_num = 0;
#pragma omp parallel for reduction(+ : non_zero_num)
  for (int64_t i = 0; i < rows; i++) {
    if (!is_zero(data + i * cols, cols)) {
      non_zero_num = non_zero_num + 1;
    }
  }
  return non_zero_num;
}

}  // namespace pten
