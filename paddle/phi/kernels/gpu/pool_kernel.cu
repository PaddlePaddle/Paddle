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

#include "paddle/phi/kernels/pool_kernel.h"

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/pool_kernel_impl.h"

PD_REGISTER_KERNEL(pool2d,
                   GPU,
                   ALL_LAYOUT,
                   phi::Pool2dKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(lp_pool2d,
                   GPU,
                   ALL_LAYOUT,
                   phi::LPPool2dKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(max_pool2d_with_index,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaxPool2dWithIndexKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}

PD_REGISTER_KERNEL(pool3d,
                   GPU,
                   ALL_LAYOUT,
                   phi::Pool3dKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(max_pool3d_with_index,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaxPool3dWithIndexKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}

PD_REGISTER_KERNEL(fractional_max_pool2d,
                   GPU,
                   ALL_LAYOUT,
                   phi::FractionalMaxPool2dKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}

PD_REGISTER_KERNEL(fractional_max_pool3d,
                   GPU,
                   ALL_LAYOUT,
                   phi::FractionalMaxPool3dKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}
