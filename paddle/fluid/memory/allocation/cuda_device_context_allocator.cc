// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/cuda_device_context_allocator.h"

#include <cuda_runtime.h>

#include "paddle/fluid/memory/allocation/cuda_device_context_allocation.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

CUDADeviceContextAllocator::CUDADeviceContextAllocator(
    const platform::CUDAPlace place)
    : place_(place) {
  platform::CUDADeviceGuard guard(place_.device);
  PADDLE_ENFORCE(cudaSuccess ==
                 cudaEventCreate(&event_, cudaEventDisableTiming));
}

CUDADeviceContextAllocator::~CUDADeviceContextAllocator() {
  if (event_) {
    platform::CUDADeviceGuard guard(place_.device);
    PADDLE_ENFORCE(cudaSuccess == cudaEventDestroy(event_));
  }
}

Allocation *CUDADeviceContextAllocator::AllocateImpl(size_t size) {
  std::lock_guard<std::mutex> lock(mtx_);
  platform::CUDADeviceGuard guard(place_.device);
  auto allocation =
      new CUDADeviceContextAllocation(memory::Alloc(place_, size));
  // Wait for the event on default stream
  PADDLE_ENFORCE(cudaSuccess == cudaEventRecord(event_));
  PADDLE_ENFORCE(cudaSuccess ==
                 cudaStreamWaitEvent(/* stream = */ 0, event_, 0));
  return allocation;
}

void CUDADeviceContextAllocator::FreeImpl(Allocation *allocation) {
  delete allocation;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
