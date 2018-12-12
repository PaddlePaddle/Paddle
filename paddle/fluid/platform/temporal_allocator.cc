// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/temporal_allocator.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace platform {
namespace alloc = memory::allocation;
#ifdef PADDLE_WITH_CUDA
TemporalAllocation::TemporalAllocation(
    alloc::AllocationPtr &&underlying_allocation, const cudaStream_t &stream)
    : Allocation(underlying_allocation->ptr(), underlying_allocation->size(),
                 underlying_allocation->place()),
      stream_(stream) {}
#endif

TemporalAllocation::TemporalAllocation(
    alloc::AllocationPtr &&underlying_allocation)
    : Allocation(underlying_allocation->ptr(), underlying_allocation->size(),
                 underlying_allocation->place()),
      underlying_allocation_(std::move(underlying_allocation)) {}

TemporalAllocator::TemporalAllocator(platform::Place place) : place_(place) {
  if (platform::is_gpu_place(place_)) {
#ifdef PADDLE_WITH_CUDA
    auto *ctx = platform::DeviceContextPool::Instance().Get(place_);
    auto *dev_ctx = static_cast<platform::CUDADeviceContext *>(ctx);
    stream_ = dev_ctx->stream();
#else
    PADDLE_THROW("Not compile with CUDA");
#endif
  }
}

bool TemporalAllocator::IsAllocThreadSafe() const { return true; }

void TemporalAllocator::Free(alloc::Allocation *allocation) {
  auto temp_allocation = dynamic_cast<TemporalAllocation *>(allocation);
  PADDLE_ENFORCE_NOT_NULL(temp_allocation);

  if (platform::is_gpu_place(temp_allocation->place())) {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE(cudaStreamSynchronize(temp_allocation->stream_));
#else
    PADDLE_THROW("Not compile with CUDA.");
#endif
  }
  delete temp_allocation;
}

alloc::Allocation *TemporalAllocator::AllocateImpl(
    size_t size, alloc::Allocator::Attr attr) {
  auto raw_allocation =
      alloc::AllocatorFacade::Instance().Alloc(place_, size, attr);
  if (platform::is_gpu_place(place_)) {
#ifdef PADDLE_WITH_CUDA
    return new TemporalAllocation(std::move(raw_allocation), stream_);
#else
    PADDLE_THROW("Not compile with CUDA");
#endif
  }
  return new TemporalAllocation(std::move(raw_allocation));
}

}  // namespace platform
}  // namespace paddle
