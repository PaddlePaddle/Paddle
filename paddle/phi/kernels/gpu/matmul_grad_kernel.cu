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

#include "paddle/phi/kernels/matmul_grad_kernel.h"

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/matmul_grad_kernel_impl.h"

PD_REGISTER_KERNEL(matmul_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(matmul_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(matmul_triple_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulTripleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(matmul_with_flatten_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulWithFlattenGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(matmul_with_flatten_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulWithFlattenDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(legacy_matmul_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LegacyMatmulGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
