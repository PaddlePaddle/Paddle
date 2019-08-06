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
    cudaEventDestroy(event_);
  }
}

void CUDADeviceContextAllocator::SetComputeStream(cudaStream_t compute_stream) {
  compute_stream_ = compute_stream;
}

Allocation *CUDADeviceContextAllocator::AllocateImpl(size_t size) {
  PADDLE_ENFORCE_NOT_NULL(
      compute_stream_,
      "Didn't set compute stream for CUDADeviceContextAllocator");
  platform::CUDADeviceGuard guard(place_.device);
  auto allocation =
      new CUDADeviceContextAllocation(memory::Alloc(place_, size));
  // Wait for the event on default stream
  PADDLE_ENFORCE(cudaSuccess == cudaEventRecord(event_, compute_stream_));
  PADDLE_ENFORCE(cudaSuccess ==
                 cudaStreamWaitEvent(compute_stream_, event_, 0));
  return allocation;
}

void CUDADeviceContextAllocator::FreeImpl(Allocation *allocation) {
  delete allocation;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
