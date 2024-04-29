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

#include "paddle/phi/kernels/lamb_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/lamb_kernel_impl.h"

PD_REGISTER_KERNEL(lamb,
                   GPU,
                   ALL_LAYOUT,
                   phi::LambKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double) {
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(4).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(5).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
}
