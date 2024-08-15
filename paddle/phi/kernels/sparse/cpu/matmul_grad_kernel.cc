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

#include "paddle/phi/kernels/sparse/matmul_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi::sparse {

// TODO(zhouwei25): implement CPU backward kernel of " CSR @ DENSE -> DENSE"
template <typename T, typename Context>
void MatmulCsrDenseGradKernel(const Context& dev_ctx UNUSED,
                              const SparseCsrTensor& x UNUSED,
                              const DenseTensor& y UNUSED,
                              const DenseTensor& dout UNUSED,
                              SparseCsrTensor* dx UNUSED,
                              DenseTensor* dy UNUSED) {
  PADDLE_THROW(common::errors::Unimplemented(
      "Not support CPU backward kernel of 'sparse.matmul' now."));
}

// TODO(zhouwei25): implement CPU kernel of " DENSE @ DENSE * CSR_MASK -> CSR"
template <typename T, typename Context>
void MaskedMatmulCsrGradKernel(const Context& dev_ctx UNUSED,
                               const DenseTensor& x UNUSED,
                               const DenseTensor& y UNUSED,
                               const SparseCsrTensor& dout UNUSED,
                               DenseTensor* dx UNUSED,
                               DenseTensor* dy UNUSED) {
  PADDLE_THROW(common::errors::Unimplemented(
      "Not support CPU backward kernel of 'sparse.masked_matmul' now."));
}

}  // namespace phi::sparse

PD_REGISTER_KERNEL(matmul_csr_dense_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MatmulCsrDenseGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(masked_matmul_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskedMatmulCsrGradKernel,
                   float,
                   double) {}
