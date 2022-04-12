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
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void ElementWiseAddCsrGradKernel(const Context& dev_ctx,
                                 const SparseCsrTensor& x,
                                 const SparseCsrTensor& y,
                                 const SparseCsrTensor& dout,
                                 SparseCsrTensor* dx,
                                 SparseCsrTensor* dy);

template <typename T, typename Context>
void ElementWiseSubtractCsrGradKernel(const Context& dev_ctx,
                                      const SparseCsrTensor& x,
                                      const SparseCsrTensor& y,
                                      const SparseCsrTensor& dout,
                                      SparseCsrTensor* dx,
                                      SparseCsrTensor* dy);

template <typename T, typename Context>
void ElementWiseMultiplyCsrGradKernel(const Context& dev_ctx,
                                      const SparseCsrTensor& x,
                                      const SparseCsrTensor& y,
                                      const SparseCsrTensor& dout,
                                      SparseCsrTensor* dx,
                                      SparseCsrTensor* dy);

template <typename T, typename Context>
void ElementWiseDivideCsrGradKernel(const Context& dev_ctx,
                                    const SparseCsrTensor& x,
                                    const SparseCsrTensor& y,
                                    const SparseCsrTensor& dout,
                                    SparseCsrTensor* dx,
                                    SparseCsrTensor* dy);

#define DEFINE_CSR_ELEMENTWISE_GRAD_KERNEL_FUNC(name)      \
  template <typename T, typename Context>                  \
  std::vector<SparseCsrTensor> ElementWise##name##CsrGrad( \
      const Context& dev_ctx,                              \
      const SparseCsrTensor& x,                            \
      const SparseCsrTensor& y,                            \
      const SparseCsrTensor& dout) {                       \
    SparseCsrTensor dx;                                    \
    SparseCsrTensor dy;                                    \
    ElementWise##name##CsrGradKernel<T, Context>(          \
        dev_ctx, x, y, dout, &dx, &dy);                    \
    return std::vector<SparseCsrTensor>{dx, dy};           \
  }

DEFINE_CSR_ELEMENTWISE_GRAD_KERNEL_FUNC(Add)
DEFINE_CSR_ELEMENTWISE_GRAD_KERNEL_FUNC(Subtract)
DEFINE_CSR_ELEMENTWISE_GRAD_KERNEL_FUNC(Multiply)
DEFINE_CSR_ELEMENTWISE_GRAD_KERNEL_FUNC(Divide)

}  // namespace sparse
}  // namespace phi
