// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/impl/cross_entropy2_kernel_impl.h"

PD_REGISTER_KERNEL(cross_entropy_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyGradientOpKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(cross_entropy_grad2,
                   GPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyGradientOpKernel2,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
}
