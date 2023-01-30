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

#include "paddle/phi/kernels/bmm_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/bmm_grad_kernel_impl.h"

PD_REGISTER_KERNEL(bmm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BmmGradKernel,
                   float,
                   double,
<<<<<<< HEAD
                   phi::dtype::float16) {}
=======
                   paddle::platform::float16) {}
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
