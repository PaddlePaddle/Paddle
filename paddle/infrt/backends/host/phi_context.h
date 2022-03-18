/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/infrt/backends/host/phi_allocator.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace infrt {
namespace backends {

class CpuPhiContext : public phi::CPUContext {
 public:
  using Base = phi::CPUContext;
  using phi::CPUContext::SetEigenDevice;

  CpuPhiContext() {
    Init();
    SetAllocator(alloc_.get());
  }

 private:
  std::unique_ptr<phi::Allocator> alloc_{std::make_unique<CpuPhiAllocator>()};
};

class GpuPhiContext : public phi::GPUContext {
 public:
  using Base = phi::GPUContext;
  using phi::GPUContext::SetStream;
  using phi::GPUContext::SetEigenDevice;
  using phi::GPUContext::SetBlasHandle;
  using phi::GPUContext::SetDnnHandle;
  using phi::GPUContext::SetSolverHandle;
  using phi::GPUContext::SetSparseHandle;
};

}  // namespace backends
}  // namespace infrt
