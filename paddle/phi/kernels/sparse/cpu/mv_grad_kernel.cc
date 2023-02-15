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

#include "paddle/phi/kernels/sparse/mv_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void MvCooGradKernel(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     const DenseTensor& vec,
                     const DenseTensor& dout,
                     SparseCooTensor* dx,
                     DenseTensor* dvec) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "Not support CPU backward kernel of 'sparse.mv' now."));
}

template <typename T, typename Context>
void MvCsrGradKernel(const Context& dev_ctx,
                     const SparseCsrTensor& x,
                     const DenseTensor& vec,
                     const DenseTensor& dout,
                     SparseCsrTensor* dx,
                     DenseTensor* dvec) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "Not support CPU backward kernel of 'sparse.mv' now."));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(
    mv_coo_grad, CPU, ALL_LAYOUT, phi::sparse::MvCooGradKernel, float, double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(
    mv_csr_grad, CPU, ALL_LAYOUT, phi::sparse::MvCsrGradKernel, float, double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
