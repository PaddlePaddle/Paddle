// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/sparse/math_grad_kernel.h"

#include "paddle/phi/kernels/activation_grad_kernel.h"
#include "paddle/phi/kernels/sparse/utils.h"

DEFINE_AND_REGISTER_SPARSE_UNARY_GRAD_KERNEL(sin_grad, SinGradKernel)

// NOTE: the following code is to bypass the restriction of Paddle
// kernel registration mechanism. Do NOT refactor them unless you
// know what you are doing.
// If you want to implement any new kernel, please follow the above
// `sin_grad`, do NOT follow the following `sqrt_grad`.
DEFINE_SPARSE_UNARY_GRAD_KERNEL(SqrtGradKernel)

PD_REGISTER_KERNEL(sparse_coo_sqrt_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooSqrtGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
PD_REGISTER_KERNEL(sparse_csr_sqrt_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCsrSqrtGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(sparse_coo_sqrt_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooSqrtGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(sparse_csr_sqrt_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCsrSqrtGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
#endif
