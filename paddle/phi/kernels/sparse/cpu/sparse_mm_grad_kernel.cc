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

#include "paddle/phi/kernels/sparse/sparse_mm_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace sparse {

// TODO(zhouwei25): implement CPU backward kernel of " CSR @ DENSE -> DENSE"
template <typename T, typename Context>
void CsrDenseMatmulGradKernel(const Context& dev_ctx,
                              const SparseCsrTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              SparseCsrTensor* dx,
                              DenseTensor* dy) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "Not support CPU backward kernel of Sparse Matmul now."));
}

// TODO(zhouwei25): implement CPU kernel of " DENSE @ DENSE * CSR_MASK -> CSR"
template <typename T, typename Context>
void CsrMaskedMatmulGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const SparseCsrTensor& dout,
                               DenseTensor* dx,
                               DenseTensor* dy) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "Not support CPU backward kernel of Matmul Mask As Sparse now."));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(csr_dense_matmul_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrDenseMatmulGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(csr_masked_mm_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrMaskedMatmulGradKernel,
                   float,
                   double) {}
