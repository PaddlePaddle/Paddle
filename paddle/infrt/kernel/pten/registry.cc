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

#include "paddle/infrt/kernel/pten/registry.h"

#include <iostream>
#include <string>

#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/kernel/pten/allocator_kernels.h"
#include "paddle/infrt/kernel/pten/context_kernels.h"
#include "paddle/infrt/kernel/pten/dense_tensor_kernels.h"
#include "paddle/infrt/kernel/pten/infershaped/pten_kernel_launcher.h"
#include "paddle/phi/include/infermeta.h"
#include "paddle/phi/include/kernels.h"
#include "paddle/phi/kernels/matmul_kernel.h"

using infrt::host_context::Attribute;

namespace infrt {
namespace kernel {

void RegisterPtenKernels(host_context::KernelRegistry* registry) {
  registry->AddKernel("pten_dt.create_allocator.cpu",
                      INFRT_KERNEL(infrt::kernel::pten::CreateCpuAllocator));
  registry->AddKernel("pten_dt.create_context.cpu",
                      INFRT_KERNEL(infrt::kernel::pten::CreateCpuContext));
  registry->AddKernel(
      "pten_dt.create_dense_tensor.cpu.f32.nchw",
      INFRT_KERNEL(infrt::kernel::pten::CreateDenseTensorCpuF32Nchw));
  registry->AddKernel("pten_dt.fill_dense_tensor.f32",
                      INFRT_KERNEL(infrt::kernel::pten::FillDenseTensorF32));
  registry->AddKernel(
      "pten.matmul.host.fp32",
      std::bind(&kernel::KernelLauncherFunc<
                    decltype(&::phi::MatmulKernel<float, ::phi::CPUContext>),
                    &::phi::MatmulKernel<float, ::phi::CPUContext>,
                    decltype(&::phi::MatmulInferMeta),
                    &::phi::MatmulInferMeta>,
                kernel::KernelLauncher<
                    decltype(&::phi::MatmulKernel<float, ::phi::CPUContext>),
                    &::phi::MatmulKernel<float, ::phi::CPUContext>,
                    decltype(&::phi::MatmulInferMeta),
                    &::phi::MatmulInferMeta>(),
                std::placeholders::_1));
}

}  // namespace kernel
}  // namespace infrt
