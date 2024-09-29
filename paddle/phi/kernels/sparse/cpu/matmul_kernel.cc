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

namespace phi::sparse {

// TODO(zhouwei25): implement CPU kernel of " CSR @ DENSE -> DENSE"
template <typename T, typename Context>
void MatmulCsrDenseKernel(const Context& dev_ctx UNUSED,
                          const SparseCsrTensor& x UNUSED,
                          const DenseTensor& y UNUSED,
                          DenseTensor* out UNUSED) {
  PADDLE_THROW(common::errors::Unimplemented(
      "Not support CPU kernel of 'sparse.matmul' now."));
}

// TODO(zhouwei25): implement CPU kernel of " DENSE @ DENSE * CSR_MASK -> CSR"
template <typename T, typename Context>
void MaskedMatmulCsrKernel(const Context& dev_ctx UNUSED,
                           const DenseTensor& x UNUSED,
                           const DenseTensor& y UNUSED,
                           const SparseCsrTensor& mask UNUSED,
                           SparseCsrTensor* out UNUSED) {
  PADDLE_THROW(common::errors::Unimplemented(
      "Not support CPU kernel of 'sparse.masked_matmul' now."));
}

}  // namespace phi::sparse

PD_REGISTER_KERNEL(matmul_csr_dense,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MatmulCsrDenseKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(masked_matmul_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskedMatmulCsrKernel,
                   float,
                   double) {}
