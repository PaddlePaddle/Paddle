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

#include "paddle/infrt/kernel/tensorrt/registry.h"

#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/kernel/tensorrt/trt_kernels.h"

namespace infrt {
namespace kernel {

void RegisterTrtKernels(host_context::KernelRegistry* registry) {
  registry->AddKernel("trt.create_engine",
                      INFRT_KERNEL(tensorrt::CreateTrtEngine));
  registry->AddKernel("trt.inspect_engine",
                      INFRT_KERNEL(tensorrt::PrintTrtLayer));
  registry->AddKernel("trt.compute", INFRT_KERNEL(tensorrt::TrtEngineCompute));
}

}  // namespace kernel
}  // namespace infrt
