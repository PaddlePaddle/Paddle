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
  registry->AddKernel("phi_dt.create_context.cpu",
                      INFRT_KERNEL(infrt::kernel::phi::CreateCPUContext));
  registry->AddKernel("phi_dt.create_dense_tensor.cpu",
                      INFRT_KERNEL(infrt::kernel::phi::CreateDenseTensor),
                      {"dims", "lod", "layout", "precision"});

  registry->AddKernel(
      "phi_dt.create_inited_dense_tensor.cpu.f32",
      INFRT_KERNEL(infrt::kernel::phi::CreateInitedDenseTensorF32),
      {"dims", "lod", "layout", "value"});

  registry->AddKernel(
      "phi_dt.create_host_inited_dense_tensor.f32",
      INFRT_KERNEL(infrt::kernel::phi::CreateHostInitedDenseTensorF32),
      {"dims", "lod", "layout", "values", "run_once"});

  registry->AddKernel("phi_dt.fill_dense_tensor.f32",
                      INFRT_KERNEL(infrt::kernel::phi::FillDenseTensorF32),
                      {"value"});

  registry->AddKernel("phi_dt.print_tensor",
                      INFRT_KERNEL(infrt::kernel::phi::PrintDenseTensor));

#ifdef INFRT_WITH_GPU
  registry->AddKernel("phi_dt.create_context.gpu",
                      INFRT_KERNEL(infrt::kernel::phi::CreateGPUContext));
  registry->AddKernel("phi_dt.create_dense_tensor.gpu",
                      INFRT_KERNEL(infrt::kernel::phi::CreateGPUDenseTensor),
                      {"dims", "lod", "layout", "precision"});
  registry->AddKernel("phi_dt.memcpy.gpu",
                      INFRT_KERNEL(infrt::kernel::phi::GpuMemCpy),
                      {"d2h"});
#endif
  registry->AddKernel("phi_dt.load_params",
                      INFRT_KERNEL(infrt::kernel::phi::LoadParams),
                      {"path"});
  registry->AddKernel("phi_dt.load_combined_params_to_gpu",
                      INFRT_KERNEL(infrt::kernel::phi::LoadCombinedParamsToGpu),
                      {"model_path", "params_path"});
  registry->AddKernel("phi_dt.load_combined_params",
                      INFRT_KERNEL(infrt::kernel::phi::LoadCombinedParams),
                      {"model_path", "params_path"});
  registry->AddKernel("phi_dt.tensor_map_get_tensor",
                      INFRT_KERNEL(infrt::kernel::phi::TensorMapGetTensor),
                      {"name"});
  registry->AddKernel("phi_dt.tensor_map_get_size",
                      INFRT_KERNEL(infrt::kernel::phi::TensorMapGetSize));
}

}  // namespace kernel
}  // namespace infrt
