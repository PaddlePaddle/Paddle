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
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {
namespace sparse {

#define DECLARE_SPARSE_ACTIVATION_KERNEL(name)                                 \
  template <typename T, typename Context>                                      \
  void SparseCoo##name##Kernel(                                                \
      const Context& dev_ctx, const SparseCooTensor& x, SparseCooTensor* out); \
                                                                               \
  template <typename T, typename Context>                                      \
  void SparseCsr##name##Kernel(                                                \
      const Context& dev_ctx, const SparseCsrTensor& x, SparseCsrTensor* out);

DECLARE_SPARSE_ACTIVATION_KERNEL(Relu)
DECLARE_SPARSE_ACTIVATION_KERNEL(Sqrt)

#undef DECLARE_SPARSE_ACTIVATION_KERNEL

template <typename T, typename Context>
SparseCooTensor SparseRelu(const Context& dev_ctx, const SparseCooTensor& x) {
  DenseTensor indices, values;
  SparseCooTensor coo(indices, values, x.dims());
  SparseCooReluKernel<T, Context>(dev_ctx, x, &coo);
  return coo;
}

template <typename T, typename Context>
SparseCooTensor SparseSqrt(const Context& dev_ctx, const SparseCooTensor& x) {
  DenseTensor indices, values;
  SparseCooTensor coo(indices, values, x.dims());
  SparseCooSqrtKernel<T, Context>(dev_ctx, x, &coo);
  return coo;
}

}  // namespace sparse
}  // namespace phi
