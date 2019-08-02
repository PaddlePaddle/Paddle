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

#pragma once
#include "paddle/fluid/memory/allocation/cuda_device_context_allocator.h"

#include <cuda_runtime.h>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

CUDADeviceContextAllocator::CUDADeviceContextAllocator(
    const platform::CUDAPlace place) place_(place) {
  cudaEventCreate(&event_);
}

CUDADeviceContextAllocator::CUDADeviceContextAllocator(
    const platform::CUDAPlace place) place_(place) {
  if (event_) {
    PADDLE_ENFORCE(cudaEventDestroy(&event_));
  }
}

Allocation *CUDADeviceContextAllocator::AllocateImpl(size_t size) {
  auto allocation =
      new CUDADeviceContextAllocation(memory::Alloc(place_, size_));
  if (event_) {
    cudaEventRecord(event_);
  }
  return allocation;
}

void CUDADeviceContextAllocator::Free(Allocation *allocation) {
  delete allocation;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
