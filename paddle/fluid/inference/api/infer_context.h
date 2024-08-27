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
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/xpu_l3_strategy.h"
#endif
#include <unordered_set>

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

#ifdef PADDLE_WITH_XPU
class InferXPUContext : public phi::XPUContext {
 public:
  explicit InferXPUContext(const phi::Place& place, int context_gm_size = -1);

  void* Alloc(phi::TensorBase* tensor,
              phi::DataType dtype,
              size_t requested_size = 0,
              bool pinned = false,
              bool fake_alloc = false) const override;

  void SetXContext(xpu::Context* x_context);

  void SetL3Info(size_t l3_size,
                 void* l3_ptr,
                 size_t l3_autotune_size,
                 const phi::Place& place);

  void L3CacheAutotune();

  void SetConvAutotuneInfo(std::string conv_autotune_file,
                           int conv_autotune_level,
                           bool conv_autotune_file_writeback,
                           const phi::Place& place);

  void SetFcAutotuneInfo(std::string fc_autotune_file,
                         int fc_autotune_level,
                         bool fc_autotune_file_writeback,
                         const phi::Place& place);
  void SetContextOption(const char* name, const char* value);

  void SetOutHolder(phi::Allocation* holder);

 private:
  size_t l3_size_{0};
  void* l3_ptr_{nullptr};
  bool l3_owned_{false};
  size_t l3_autotune_size_{0};
  mutable std::vector<phi::XPUL3CacheBlock*> l3_blocks_;
  mutable std::unordered_map<phi::Allocation*, phi::XPUL3CacheBlock*>
      holder_l3_blocks_;
  mutable std::unordered_map<phi::Allocation*,
                             std::pair<phi::Allocation*, bool>>
      holder_map_;

  mutable std::unordered_set<phi::Allocation*> output_holder_set_;
  phi::XPUL3Planner l3_plan_;
};
#endif
}  // namespace paddle
