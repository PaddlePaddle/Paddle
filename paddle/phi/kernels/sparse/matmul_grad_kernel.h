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

namespace phi {
namespace sparse {

// TODO(zhouwei25): implement Backward of " COO @ COO -> COO"
template <typename T, typename Context>
void MatmulCooCooGradKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const SparseCooTensor& y,
                            const SparseCooTensor& dout,
                            SparseCooTensor* dx,
                            SparseCooTensor* dy);

// Backward of " COO @ DENSE -> DENSE"
template <typename T, typename Context>
void MatmulCooDenseGradKernel(const Context& dev_ctx,
                              const SparseCooTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              SparseCooTensor* dx,
                              DenseTensor* dy);

// TODO(zhouwei25): implement Backward of " CSR @ CSR -> CSR"
template <typename T, typename Context>
void MatmulCsrCsrGradKernel(const Context& dev_ctx,
                            const SparseCsrTensor& x,
                            const SparseCsrTensor& y,
                            const SparseCsrTensor& dout,
                            SparseCsrTensor* dx,
                            SparseCsrTensor* dy);

/* Backward of "CSR @ DENSE -> DENSE" */
template <typename T, typename Context>
void MatmulCsrDenseGradKernel(const Context& dev_ctx,
                              const SparseCsrTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              SparseCsrTensor* dx,
                              DenseTensor* dy);

/* Backward of "DENSE @ DENSE * CSR_MASK -> CSR" */
template <typename T, typename Context>
void MaskedMatmulCsrGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const SparseCsrTensor& dout,
                               DenseTensor* dx,
                               DenseTensor* dy);

}  // namespace sparse
}  // namespace phi
