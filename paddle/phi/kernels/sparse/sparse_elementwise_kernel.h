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
#include "paddle/phi/core/sparse_csr_tensor.h"
//#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {
namespace sparse {

#define CSR_ELEMENTWISE_KERNEL_NAME(name) ElementWise##name##CsrKernel

#define DEFINE_CSR_ELEMENTWISE_KERNEL_HEAD(name)              \
  template <typename T, typename Context>                     \
  void ElementWise##name##CsrKernel(const Context& dev_ctx,   \
                                    const SparseCsrTensor& x, \
                                    const SparseCsrTensor& y, \
                                    SparseCsrTensor* out);

#define DEFINE_CSR_ELEMENTWISE_KERNEL_FUNC(name)                            \
  template <typename T, typename Context>                                   \
  SparseCsrTensor ElementWise##name##Csr(const Context& dev_ctx,            \
                                         const SparseCsrTensor& x,          \
                                         const SparseCsrTensor& y) {        \
    DenseTensor non_zero_crows;                                             \
    DenseTensor non_zero_cols;                                              \
    DenseTensor non_zero_elements;                                          \
    SparseCsrTensor out(                                                    \
        non_zero_crows, non_zero_cols, non_zero_elements, x.dims());        \
    CSR_ELEMENTWISE_KERNEL_NAME(name)<T, Context>(dev_ctx, x, y, &out); \
    return out;                                                             \
  }

DEFINE_CSR_ELEMENTWISE_KERNEL_HEAD(Add)
DEFINE_CSR_ELEMENTWISE_KERNEL_FUNC(Add)

DEFINE_CSR_ELEMENTWISE_KERNEL_HEAD(Subtract)
DEFINE_CSR_ELEMENTWISE_KERNEL_FUNC(Subtract)

DEFINE_CSR_ELEMENTWISE_KERNEL_HEAD(Multiply)
DEFINE_CSR_ELEMENTWISE_KERNEL_FUNC(Multiply)

DEFINE_CSR_ELEMENTWISE_KERNEL_HEAD(Divide)
DEFINE_CSR_ELEMENTWISE_KERNEL_FUNC(Divide)

// template <typename Functor, typename T, typename Context>
// void ElementWiseCsrKernel(const Context& dev_ctx,
//                           const SparseCsrTensor& x,
//                           const SparseCsrTensor& y,
//                           SparseCsrTensor* out);

}  // namespace sparse
}  // namespace phi
