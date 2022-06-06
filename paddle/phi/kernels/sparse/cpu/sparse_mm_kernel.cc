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

#include "paddle/phi/kernels/sparse/sparse_mm_kernel.h"

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
      "Only support GPU kernel of Sparse Matmul now."));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(csr_dense_matmul,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrDenseMatmulKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
