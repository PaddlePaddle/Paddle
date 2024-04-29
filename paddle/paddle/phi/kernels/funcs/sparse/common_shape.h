/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <stdint.h>

#include "paddle/common/ddim.h"

namespace phi {
namespace funcs {
namespace sparse {

inline const DDim InferDenseDims(const DDim& x_dims,
                                 const int64_t sparse_dim,
                                 const int64_t non_zero_num) {
  auto dense_dim = x_dims.size() - sparse_dim;
  DDim values_dims;
  if (dense_dim > 0) {
    std::vector<int64_t> dense_dim_vec(dense_dim + 1);
    dense_dim_vec[0] = non_zero_num;
    memcpy(&dense_dim_vec[1],
           x_dims.Get() + sparse_dim,
           dense_dim * sizeof(x_dims[0]));
    values_dims = common::make_ddim(dense_dim_vec);
  } else {
    values_dims = common::make_ddim({non_zero_num});
  }
  return values_dims;
}

template <typename IntT>
inline const IntT HOSTDEVICE IndicesToIndex(const IntT* indices,
                                            const IntT* sparse_offsets,
                                            const int64_t non_zero_num,
                                            const int64_t sparse_dim,
                                            const int i) {
  IntT index = 0;
  for (IntT j = 0; j < sparse_dim; j++) {
    index += indices[j * non_zero_num + i] * sparse_offsets[j];
  }
  return index;
}

template <typename IntT>
inline void HOSTDEVICE FlattenIndices(const IntT* indices,
                                      const IntT* sparse_offsets,
                                      const int64_t non_zero_num,
                                      const int64_t sparse_dim,
                                      const int start,
                                      const int stride,
                                      IntT* out) {
  for (int i = start; i < non_zero_num; i += stride) {
    out[i] =
        IndicesToIndex(indices, sparse_offsets, non_zero_num, sparse_dim, i);
  }
}

// 1. indices.dims().size() == 2
template <typename IntT>
inline void CalcOffsetsPerDim(const DDim& dims,
                              const int64_t sparse_dim,
                              std::vector<IntT>* offsets) {
  IntT offset = 1;
  for (IntT i = sparse_dim - 1; i >= 0; i--) {
    (*offsets)[i] = offset;
    offset *= dims[i];
  }
}

}  // namespace sparse
}  // namespace funcs
}  // namespace phi
