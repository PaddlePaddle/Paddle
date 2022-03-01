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

#include "paddle/infrt/kernel/phi/registry.h"

#include <iostream>
#include <string>

#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/kernel/phi/allocator_kernels.h"
#include "paddle/infrt/kernel/phi/context_kernels.h"
#include "paddle/infrt/kernel/phi/dense_tensor_kernels.h"
#include "paddle/infrt/kernel/phi/infershaped/phi_kernel_launcher.h"
#include "paddle/phi/include/infermeta.h"
#include "paddle/phi/include/kernels.h"
#include "paddle/phi/kernels/matmul_kernel.h"

using infrt::host_context::Attribute;

namespace infrt {
namespace kernel {

void RegisterPhiKernels(host_context::KernelRegistry* registry) {
  registry->AddKernel("phi_dt.create_allocator.cpu",
                      INFRT_KERNEL(infrt::kernel::phi::CreateCpuAllocator));
  registry->AddKernel("phi_dt.create_context.cpu",
                      INFRT_KERNEL(infrt::kernel::phi::CreateCpuContext));
  registry->AddKernel(
      "phi_dt.create_dense_tensor.cpu.f32.nchw",
      INFRT_KERNEL(infrt::kernel::phi::CreateDenseTensorCpuF32Nchw));
  registry->AddKernel("phi_dt.fill_dense_tensor.f32",
                      INFRT_KERNEL(infrt::kernel::phi::FillDenseTensorF32));
  registry->AddKernel(
      "phi_dt.fake_phi_kernel",
      std::bind(&KernelLauncherFunc<decltype(&FakePhiKernel),
                                    &FakePhiKernel,
                                    decltype(&FakePhiInferShape),
                                    &FakePhiInferShape>,
                KernelLauncher<decltype(&FakePhiKernel),
                               &FakePhiKernel,
                               decltype(&FakePhiInferShape),
                               &FakePhiInferShape>(),
                std::placeholders::_1));
}

}  // namespace kernel
}  // namespace infrt
