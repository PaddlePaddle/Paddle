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

// TODO(zhouwei25): implement Backward of " COO + COO @ COO -> COO"
template <typename T, typename Context>
void AddmmCooCooGradKernel(const Context& dev_ctx,
                           const SparseCooTensor& input,
                           const SparseCooTensor& x,
                           const SparseCooTensor& y,
                           const SparseCooTensor& dout,
                           float alpha,
                           float beta,
                           SparseCooTensor* dinput,
                           SparseCooTensor* dx,
                           SparseCooTensor* dy);

// Backward of "DENSE + COO @ DENSE -> DENSE"
template <typename T, typename Context>
void AddmmCooDenseGradKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             const SparseCooTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& dout,
                             float alpha,
                             float beta,
                             DenseTensor* dinput,
                             SparseCooTensor* dx,
                             DenseTensor* dy);

// TODO(zhouwei25): implement Backward of " CSR + CSR @ CSR -> CSR"
template <typename T, typename Context>
void AddmmCsrCsrGradKernel(const Context& dev_ctx,
                           const SparseCsrTensor& input,
                           const SparseCsrTensor& x,
                           const SparseCsrTensor& y,
                           const SparseCsrTensor& dout,
                           float alpha,
                           float beta,
                           SparseCsrTensor* dinput,
                           SparseCsrTensor* dx,
                           SparseCsrTensor* dy);

/* Backward of "DENSE + CSR @ DENSE -> DENSE" */
template <typename T, typename Context>
void AddmmCsrDenseGradKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             const SparseCsrTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& dout,
                             float alpha,
                             float beta,
                             DenseTensor* dinput,
                             SparseCsrTensor* dx,
                             DenseTensor* dy);

}  // namespace sparse
}  // namespace phi
