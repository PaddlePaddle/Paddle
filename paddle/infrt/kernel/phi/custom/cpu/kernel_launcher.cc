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

#include "paddle/infrt/kernel/phi/custom/cpu/kernel_launcher.h"

#include "paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launchers.h"
#include "paddle/infrt/kernel/phi/infershaped/phi_kernel_launcher.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/include/infermeta.h"
#include "paddle/phi/include/kernels.h"
#include "paddle/phi/infermeta/generated.h"

namespace infrt {
namespace kernel {

void RegisterCpuKernelLaunchers(host_context::KernelRegistry* registry) {
  registry->AddKernel(
      "phi_cpu.fc.float32.any",
      &KernelLauncherFunc<decltype(&FcKernel<float, ::phi::CPUContext>),
                          &FcKernel<float, ::phi::CPUContext>,
                          decltype(&FcInferMeta),
                          &FcInferMeta>);
  registry->AddKernel(
      "phi_cpu.fc.float64.any",
      &KernelLauncherFunc<decltype(&FcKernel<double, ::phi::CPUContext>),
                          &FcKernel<double, ::phi::CPUContext>,
                          decltype(&FcInferMeta),
                          &FcInferMeta>);
}
}  // namespace kernel
}  // namespace infrt
