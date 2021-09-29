// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace framework {

static void GetTensors(Variable* var, std::unordered_set<Tensor*>* tensor_set) {
  if (var->IsType<LoDTensor>() && var->Get<LoDTensor>().IsInitialized()) {
    tensor_set->insert(var->GetMutable<LoDTensor>());
  } else if (var->IsType<SelectedRows>() &&
             var->Get<SelectedRows>().value().IsInitialized()) {
    tensor_set->insert(var->GetMutable<SelectedRows>()->mutable_value());
  } else if (var->IsType<LoDTensorArray>()) {
    auto* tensor_arr = var->GetMutable<LoDTensorArray>();
    for (auto& t : *tensor_arr) {
      if (t.IsInitialized()) {
        tensor_set->insert(&t);
      }
    }
  }
}

static std::pair<size_t, size_t> GetTensorMemorySize(
    const std::vector<Variable*>& var_list) {
  std::unordered_set<Tensor*> tensor_set;
  for (auto* var : var_list) {
    GetTensors(var, &tensor_set);
  }
  size_t host_memory_bytes = 0;
  size_t device_memory_bytes = 0;
  std::unordered_set<memory::Allocation*> allocation_set;
  for (auto* tensor : tensor_set) {
    auto allocation = tensor->Holder().get();
    if (!allocation_set.count(allocation)) {
      allocation_set.insert(allocation);
      if (platform::is_cuda_pinned_place(tensor->place()) ||
          platform::is_cpu_place(tensor->place())) {
        VLOG(3) << "found host memory : " << allocation->size();
        host_memory_bytes += allocation->size();
      } else {
        VLOG(3) << "found device memory : " << allocation->size();
        device_memory_bytes += allocation->size();
      }
    }
  }
  return {host_memory_bytes, device_memory_bytes};
}

struct CostInfo {
  double total_time{0.};          // ms
  size_t device_memory_bytes{0};  // total allocated memory size
};

class InterpreterProfiler {
 public:
  void Start() { timer_.Start(); }

  void Pause() {
    timer_.Pause();
    cost_info_.total_time += timer_.ElapsedMS();
  }

  void Reset() {
    timer_.Reset();
    cost_info_.total_time = 0.;
    cost_info_.device_memory_bytes = 0;
  }

  void TotalCUDAAllocatedMemorySize(const platform::Place& place) {
    if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      auto cuda_place = BOOST_GET_CONST(platform::CUDAPlace, place);
      cost_info_.device_memory_bytes =
          platform::RecordedCudaMallocSize(cuda_place.device);
#endif
    }
  }

  const CostInfo& GetCostInfo() const { return cost_info_; }

 private:
  platform::Timer timer_;
  CostInfo cost_info_;
};
}  // namespace framework
}  // namespace paddle
