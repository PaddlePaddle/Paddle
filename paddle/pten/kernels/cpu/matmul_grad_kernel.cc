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

#include "paddle/pten/kernels/matmul_grad_kernel.h"

#include "paddle/pten/common/complex.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/pten/kernels/impl/matmul_grad_kernel_impl.h"

PT_REGISTER_KERNEL(matmul_grad,
                   CPU,
                   ALL_LAYOUT,
                   pten::MatmulGradKernel,
                   float,
                   double,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}

PT_REGISTER_KERNEL(matmul_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   pten::MatmulDoubleGradKernel,
                   float,
                   double,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}

PT_REGISTER_KERNEL(matmul_triple_grad,
                   CPU,
                   ALL_LAYOUT,
                   pten::MatmulTripleGradKernel,
                   float,
                   double,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}
