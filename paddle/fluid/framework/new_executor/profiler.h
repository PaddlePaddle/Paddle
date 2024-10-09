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
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/timer.h"

namespace paddle {
namespace framework {
namespace interpreter {
struct CostInfo {
  double total_time{0.};          // ms
  size_t device_memory_bytes{0};  // total allocated memory size
};

class ProfilerGuard {
 public:
  ProfilerGuard(const phi::Place& place, CostInfo* cost_info)
      : place_(place), cost_info_(cost_info) {
    timer_.Start();
  }

  ~ProfilerGuard() {
    timer_.Pause();
    cost_info_->total_time += timer_.ElapsedMS();
    TotalCUDAAllocatedMemorySize(place_);
  }

 private:
  void TotalCUDAAllocatedMemorySize(const phi::Place& place) {
    if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      auto cuda_place = place;
      cost_info_->device_memory_bytes =
          platform::RecordedGpuMallocSize(cuda_place.device);
#endif
    }
  }

  const phi::Place& place_;
  CostInfo* cost_info_;
  platform::Timer timer_;
};

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
