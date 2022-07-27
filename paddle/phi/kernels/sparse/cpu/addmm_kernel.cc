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

#include "paddle/phi/kernels/sparse/addmm_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
namespace phi {
namespace sparse {

/* DENSE + COO @ DENSE -> DENSE */
template <typename T, typename Context>
void AddmmCooDenseKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const SparseCooTensor& x,
                         const DenseTensor& y,
                         float alpha,
                         float beta,
                         DenseTensor* out) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "Not support CPU kernel of 'sparse.addmm' now."));
}

/* DENSE + CSR @ DENSE -> DENSE */
template <typename T, typename Context>
void AddmmCsrDenseKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const SparseCsrTensor& x,
                         const DenseTensor& y,
                         float alpha,
                         float beta,
                         DenseTensor* out) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "Not support CPU kernel of 'sparse.addmm' now."));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(addmm_coo_dense,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::AddmmCooDenseKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(addmm_csr_dense,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::AddmmCsrDenseKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
