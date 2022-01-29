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

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/sparse_coo_tensor.h"
#include "paddle/pten/core/sparse_csr_tensor.h"
#include "paddle/pten/kernels/empty_kernel.h"

namespace pten {

inline const DDim InferDenseDims(const DDim& x_dims,
                                 const int64_t sparse_dim,
                                 const int64_t non_zero_num) {
  auto dense_dim = x_dims.size() - sparse_dim;
  DDim values_dims;
  if (dense_dim) {
    std::vector<int64_t> dense_dim_vec(dense_dim + 1);
    dense_dim_vec[0] = non_zero_num;
    memcpy(&dense_dim_vec[1],
           x_dims.Get() + sparse_dim,
           dense_dim * sizeof(x_dims[0]));
    values_dims = pten::framework::make_ddim(dense_dim_vec);
  } else {
    values_dims = pten::framework::make_ddim({non_zero_num});
  }
  return values_dims;
}

template <typename T, typename Context>
void DenseToSparseCooKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const int64_t sparse_dim,
                            SparseCooTensor* out);

template <typename T, typename Context>
SparseCooTensor DenseToSparseCoo(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const int64_t sparse_dim) {
  DenseTensor indices = pten::Empty<T, Context>(dev_ctx);
  DenseTensor values = pten::Empty<T, Context>(dev_ctx);
  SparseCooTensor coo(indices, values, x.dims());
  DenseToSparseCooKernel<T, Context>(dev_ctx, x, sparse_dim, &coo);
  return coo;
}

}  // namespace pten
