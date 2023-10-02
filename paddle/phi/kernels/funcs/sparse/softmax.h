/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace funcs {
namespace sparse {

template <typename IntT>
inline void GetPoolsSoftmax(const DenseTensor& indices,
                            const std::vector<IntT>& sizes,
                            const int dim,
                            std::map<IntT, std::vector<IntT>>* pools) {
  auto ndim = indices.dims()[0];
  auto nnz = indices.dims()[1];
  std::vector<IntT> strides(ndim, 1);

  if (ndim > 1) {
    for (IntT i = ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * (i + 1 == dim ? 1 : sizes[i + 1]);
    }
  }

  auto* indices_data = indices.data<IntT>();
  for (IntT i = 0; i < nnz; i++) {
    IntT pool_index = 0;
    for (IntT j = 0; j < ndim; j++) {
      if (j == dim) continue;
      pool_index += strides[j] * indices_data[j * nnz + i];
    }

    if (pools->find(pool_index) == pools->end()) {
      std::vector<IntT> vec;
      (*pools)[pool_index] = vec;
    }
    (*pools)[pool_index].push_back(i);
  }
}

template <typename IntT>
inline std::vector<IntT> GetOffsets(const DenseTensor& indices,
                                    const std::vector<IntT>& sizes,
                                    const int dim) {
  auto ndim = indices.dims()[0];
  auto nnz = indices.dims()[1];
  std::vector<IntT> offsets(nnz);
  std::vector<IntT> strides(ndim, 1);
  auto indices_ptr = indices.data<IntT>();

  if (ndim > 1) {
    for (IntT i = ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
  }

  for (int i = 0; i < nnz; i++) {
    IntT acc = 0;
    for (int j = 0; j < ndim; j++) {
      auto indices_cur = indices_ptr + j * nnz + i;
      auto stride = strides[j];
      if (j != dim) {
        acc += stride * (*indices_cur);
      }
    }
    offsets[i] = acc;
  }

  return offsets;
}

}  // namespace sparse
}  // namespace funcs
}  // namespace phi
