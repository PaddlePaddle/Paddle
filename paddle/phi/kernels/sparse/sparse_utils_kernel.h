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

#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void DenseToSparseCooKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const int64_t sparse_dim,
                            SparseCooTensor* out);

template <typename T, typename Context>
SparseCooTensor DenseToSparseCoo(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const int64_t sparse_dim) {
  DenseTensor indices;
  DenseTensor values;
  SparseCooTensor coo(indices, values, x.dims());
  DenseToSparseCooKernel<T, Context>(dev_ctx, x, sparse_dim, &coo);
  return coo;
}

template <typename T, typename Context>
void SparseCsrToCooKernel(const Context& dev_ctx,
                          const SparseCsrTensor& x,
                          SparseCooTensor* out);

template <typename T, typename Context>
SparseCooTensor SparseCsrToCoo(const Context& dev_ctx,
                               const SparseCsrTensor& x) {
  DenseTensor indices;
  DenseTensor values;
  SparseCooTensor coo(indices, values, x.dims());
  SparseCsrToCooKernel<T, Context>(dev_ctx, x, &coo);
  return coo;
}

template <typename T, typename Context>
void SparseCooToCsrKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          SparseCsrTensor* out);

template <typename T, typename Context>
SparseCsrTensor SparseCooToCsr(const Context& dev_ctx,
                               const SparseCooTensor& x) {
  DenseTensor non_zero_crows;
  DenseTensor non_zero_cols;
  DenseTensor non_zero_elements;
  SparseCsrTensor csr(
      non_zero_crows, non_zero_cols, non_zero_elements, x.dims());
  SparseCooToCsrKernel<T, Context>(dev_ctx, x, &csr);
  return csr;
}

template <typename T, typename Context>
void DenseToSparseCsrKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            SparseCsrTensor* out) {
  const auto& x_dims = x.dims();
  bool valid = x_dims.size() == 2 || x_dims.size() == 3;
  PADDLE_ENFORCE_EQ(valid,
                    true,
                    phi::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D or 3-D Tensor."));
  const int64_t sparse_dim = x_dims.size() == 2 ? 2 : 3;
  DenseTensor indices;
  DenseTensor values;
  SparseCooTensor coo(indices, values, x.dims());
  DenseToSparseCooKernel<T, Context>(dev_ctx, x, sparse_dim, &coo);
  SparseCooToCsrKernel<T, Context>(dev_ctx, coo, out);
}

template <typename T, typename Context>
SparseCsrTensor DenseToSparseCsr(const Context& dev_ctx, const DenseTensor& x) {
  DenseTensor non_zero_crows;
  DenseTensor non_zero_cols;
  DenseTensor non_zero_elements;
  SparseCsrTensor csr(
      non_zero_crows, non_zero_cols, non_zero_elements, x.dims());
  DenseToSparseCsrKernel<T, Context>(dev_ctx, x, &csr);
  return csr;
}

template <typename T, typename Context>
void SparseCooToDenseKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            DenseTensor* out);

template <typename T, typename Context>
DenseTensor SparseCooToDense(const Context& dev_ctx, const SparseCooTensor& x) {
  DenseTensorMeta meta(x.dtype(), x.dims(), x.layout());
  DenseTensor dense = phi::Empty(dev_ctx, std::move(meta));
  SparseCooToDenseKernel<T, Context>(dev_ctx, x, &dense);
  return dense;
}

template <typename T, typename Context>
void SparseCsrToDenseKernel(const Context& dev_ctx,
                            const SparseCsrTensor& x,
                            DenseTensor* out) {
  DenseTensor indices;
  DenseTensor values;
  SparseCooTensor coo(indices, values, x.dims());
  SparseCsrToCooKernel<T, Context>(dev_ctx, x, &coo);
  SparseCooToDenseKernel<T, Context>(dev_ctx, coo, out);
}

template <typename T, typename Context>
DenseTensor SparseCsrToDense(const Context& dev_ctx, const SparseCsrTensor& x) {
  DenseTensorMeta meta(x.dtype(), x.dims(), x.layout());
  DenseTensor dense = phi::Empty(dev_ctx, std::move(meta));
  SparseCsrToDenseKernel<T, Context>(dev_ctx, x, &dense);
  return dense;
}

}  // namespace sparse
}  // namespace phi
