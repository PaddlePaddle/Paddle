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

#include "paddle/phi/kernels/sparse/matmul_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace sparse {

// TODO(zhouwei25): implement CPU kernel of " CSR @ DENSE -> DENSE"
template <typename T, typename Context>
void CsrDenseMatmulKernel(const Context& dev_ctx,
                          const SparseCsrTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "Not support CPU kernel of 'sparse.matmul' now."));
}

// TODO(zhouwei25): implement CPU kernel of " DENSE @ DENSE * CSR_MASK -> CSR"
template <typename T, typename Context>
void CsrMaskedMatmulKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           const SparseCsrTensor& mask,
                           SparseCsrTensor* out) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "Not support CPU kernel of 'sparse.masked_matmul' now."));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(csr_dense_matmul,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrDenseMatmulKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(csr_masked_matmul,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrMaskedMatmulKernel,
                   float,
                   double) {}
