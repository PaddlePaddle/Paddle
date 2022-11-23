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

#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/sparse/elementwise_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/infermeta/binary.h"

namespace phi {
namespace sparse {

#define DEFINE_ELEMENTWISE_KERNEL_HEAD(name)          \
  DEFINE_ELEMENTWISE_KERNEL_HEAD_WITH_TYPE(name, Csr) \
                                                      \
  DEFINE_ELEMENTWISE_KERNEL_HEAD_WITH_TYPE(name, Coo)

#define DEFINE_ELEMENTWISE_KERNEL_FUNC(name) \
  DEFINE_CSR_ELEMENTWISE_KERNEL_FUNC(name)   \
                                             \
  DEFINE_COO_ELEMENTWISE_KERNEL_FUNC(name)

#define DEFINE_ELEMENTWISE_KERNEL_HEAD_WITH_TYPE(name, type)          \
  template <typename T, typename Context>                             \
  void ElementWise##name##type##Kernel(const Context& dev_ctx,        \
                                       const Sparse##type##Tensor& x, \
                                       const Sparse##type##Tensor& y, \
                                       Sparse##type##Tensor* out);

#define DEFINE_CSR_ELEMENTWISE_KERNEL_FUNC(name)                     \
  template <typename T, typename Context>                            \
  SparseCsrTensor ElementWise##name##Csr(const Context& dev_ctx,     \
                                         const SparseCsrTensor& x,   \
                                         const SparseCsrTensor& y) { \
    DenseTensor crows;                                               \
    DenseTensor cols;                                                \
    DenseTensor values;                                              \
    SparseCsrTensor out(crows, cols, values, x.dims());              \
    MetaTensor meta_out(out);                                        \
    phi::ElementwiseInferMeta(x, y, &meta_out);                      \
    ElementWise##name##CsrKernel<T, Context>(dev_ctx, x, y, &out);   \
    return out;                                                      \
  }

#define DEFINE_COO_ELEMENTWISE_KERNEL_FUNC(name)                     \
  template <typename T, typename Context>                            \
  SparseCooTensor ElementWise##name##Coo(const Context& dev_ctx,     \
                                         const SparseCooTensor& x,   \
                                         const SparseCooTensor& y) { \
    DenseTensor indices;                                             \
    DenseTensor values;                                              \
    SparseCooTensor out(indices, values, x.dims());                  \
    MetaTensor meta_out(out);                                        \
    phi::ElementwiseInferMeta(x, y, &meta_out);                      \
    ElementWise##name##CooKernel<T, Context>(dev_ctx, x, y, &out);   \
    return out;                                                      \
  }

DEFINE_ELEMENTWISE_KERNEL_HEAD(Add)
DEFINE_ELEMENTWISE_KERNEL_HEAD(Subtract)
DEFINE_ELEMENTWISE_KERNEL_HEAD(Multiply)
DEFINE_ELEMENTWISE_KERNEL_HEAD(Divide)

DEFINE_ELEMENTWISE_KERNEL_FUNC(Add)
DEFINE_ELEMENTWISE_KERNEL_FUNC(Subtract)
DEFINE_ELEMENTWISE_KERNEL_FUNC(Multiply)
DEFINE_ELEMENTWISE_KERNEL_FUNC(Divide)

template <typename T, typename Context>
void ElementWiseAddDenseKernel(const Context& dev_ctx,
                               const SparseCooTensor& x,
                               const DenseTensor& y,
                               SparseCooTensor* out) {
  // TODO(zhangkaiuo): to support universal sparse + dense
  if (y.dims().size() == 1 && y.dims()[0] == x.dims()[x.dims().size() - 1]) {
    EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);
    phi::AddKernel<T, Context>(dev_ctx, x.values(), y, out->mutable_values());
    out->SetIndicesDict(x.GetIndicesDict());
  } else {
    PADDLE_THROW(
        errors::Unimplemented("Not support Sparse + Dense in GPU mode"));
  }
}

}  // namespace sparse
}  // namespace phi
