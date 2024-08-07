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

#include "paddle/phi/kernels/pool_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/pool_grad_kernel_impl.h"

PD_REGISTER_KERNEL(
    pool2d_grad, CPU, ALL_LAYOUT, phi::Pool2dGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    lp_pool2d_grad, CPU, ALL_LAYOUT, phi::LPPool2dGradKernel, float, double) {}
PD_REGISTER_KERNEL(pool2d_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::Pool2dDoubleGradKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(max_pool2d_with_index_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaxPool2dWithIndexGradKernel,
                   float,
                   double) {
  kernel->InputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}

PD_REGISTER_KERNEL(
    pool3d_grad, CPU, ALL_LAYOUT, phi::Pool3dGradKernel, float, double) {}
PD_REGISTER_KERNEL(max_pool3d_with_index_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaxPool3dWithIndexGradKernel,
                   float,
                   double) {
  kernel->InputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}

PD_REGISTER_KERNEL(fractional_max_pool2d_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::FractionalMaxPool2dGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}

PD_REGISTER_KERNEL(fractional_max_pool3d_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::FractionalMaxPool3dGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}
