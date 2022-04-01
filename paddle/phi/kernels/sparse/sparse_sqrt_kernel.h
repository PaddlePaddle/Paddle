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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SqrtCsrKernel(const Context& dev_ctx,
                   const SparseCsrTensor& input,
                   SparseCsrTensor* out);

template <typename T, typename Context>
SparseCsrTensor Sqrt(const Context& dev_ctx, const SparseCsrTensor& x) {
  DenseTensor crows;
  DenseTensor cols;
  DenseTensor values;
  SparseCsrTensor out(crows, cols, values, x.dims());
  SqrtCsrKernel<T, Context>(dev_ctx, x, &out);

  return out;
}

template <typename T, typename Context>
SparseCooTensor Sqrt(const Context& dev_ctx, const SparseCooTensor& x) {
  DenseTensor crows;
  DenseTensor cols;
  DenseTensor values;
  SparseCsrTensor csr_x = SparseCooToCsr<T>(dev_ctx, x);
  SparseCsrTensor csr_out(crows, cols, values, x.dims());
  SqrtCsrKernel<T, Context>(dev_ctx, csr_x, &csr_out);
  auto out = SparseCsrToCoo<T>(dev_ctx, csr_out);

  return out;
}

}  // namespace sparse
}  // namespace phi
