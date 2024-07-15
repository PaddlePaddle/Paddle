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

#include "paddle/phi/kernels/sparse/mask_grad_kernel.h"
#include "paddle/phi/kernels/sparse/mask_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

PD_REGISTER_KERNEL(mask_as_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskAsCooGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(mask_as_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskAsCsrGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
