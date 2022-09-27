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
#pragma once

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/place.h"

namespace paddle {

class InferCPUContext : public phi::CPUContext {
 public:
  using phi::CPUContext::SetEigenDevice;
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
class InferGPUContext : public phi::GPUContext {
 public:
  explicit InferGPUContext(const phi::Place& place);
  using phi::GPUContext::SetBlasHandle;
  using phi::GPUContext::SetBlasTensorCoreHandle;
  using phi::GPUContext::SetBlasTF32Handle;
  using phi::GPUContext::SetDnnHandle;
  using phi::GPUContext::SetEigenDevice;
  using phi::GPUContext::SetSolverHandle;
  using phi::GPUContext::SetSparseHandle;
  using phi::GPUContext::SetStream;
  // using phi::GPUContext::SetDnnWorkspaceHandle;
  using phi::GPUContext::SetComputeCapability;
  using phi::GPUContext::SetDriverVersion;
  using phi::GPUContext::SetMaxGridDimSize;
  using phi::GPUContext::SetMaxThreadsPerBlock;
  using phi::GPUContext::SetMaxThreadsPerMultiProcessor;
  using phi::GPUContext::SetMultiProcessors;
  using phi::GPUContext::SetRuntimeVersion;
};
#endif
}  // namespace paddle
