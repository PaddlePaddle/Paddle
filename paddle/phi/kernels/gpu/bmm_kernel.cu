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

#include "paddle/phi/kernels/bmm_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/bmm_kernel_impl.h"

<<<<<<< HEAD
PD_REGISTER_KERNEL(bmm,
                   GPU,
                   ALL_LAYOUT,
                   phi::BmmKernel,
                   float,
                   double,
                   paddle::platform::float16) {}
=======
PD_REGISTER_KERNEL(
    bmm, GPU, ALL_LAYOUT, phi::BmmKernel, float, double, phi::dtype::float16) {}
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
