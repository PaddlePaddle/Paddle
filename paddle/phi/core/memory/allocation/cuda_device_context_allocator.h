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

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/device_context.h"

namespace paddle {
namespace memory {
namespace allocation {

/**
 * GPUContextAllocation is a wrapper of the underbeneath allocation.
 * GPUContextAllocation adds a CUDA stream callback for the underbeneath
 * allocation so that GPUContextAllocation can be used in a CUDA stream
 * which deletes allocation in the callback.
 */
class GPUContextAllocation : public Allocation {
 public:
  explicit GPUContextAllocation(DecoratedAllocationPtr allocation)
      : Allocation(allocation->ptr(),
                   allocation->base_ptr(),
                   allocation->size(),
                   allocation->place()),
        underlying_allocation_(std::move(allocation)) {}

  ~GPUContextAllocation() {
    PADDLE_WARN_NOT_NULL(
        dev_ctx_,
        common::errors::PreconditionNotMet(
            "Device context is not set for GPUContextAllocation"));

    auto *p_allocation = underlying_allocation_.release();
    VLOG(4) << "Adding callback to delete GPUContextAllocation at "
            << p_allocation;
    dev_ctx_->AddStreamCallback([p_allocation] {
      VLOG(4) << "Delete GPUContextAllocation at " << p_allocation;
      Allocator::AllocationDeleter(p_allocation);
    });
  }

  void SetGPUContext(const phi::GPUContext *dev_ctx) { dev_ctx_ = dev_ctx; }

 private:
  DecoratedAllocationPtr underlying_allocation_;
  const phi::GPUContext *dev_ctx_{nullptr};
};

/**
 * GPUContextAllocator will allocate a GPUContextAllocation
 * after waiting for a self-created event on the default stream. It does so to
 * let the non-default stream be able to allocate GPU memory which will be
 * released by stream callback
 */
class GPUContextAllocator : public Allocator {
 public:
  explicit GPUContextAllocator(phi::GPUPlace place, gpuStream_t default_stream)
      : place_(place), default_stream_(default_stream) {
    platform::CUDADeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        hipEventCreateWithFlags(&event_, hipEventDisableTiming));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventCreate(&event_, cudaEventDisableTiming));
#endif
  }

  ~GPUContextAllocator() {
    if (event_) {
      platform::CUDADeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_HIP

      PADDLE_WARN_GPU_SUCCESS(hipEventDestroy(event_));
#else
      PADDLE_WARN_GPU_SUCCESS(cudaEventDestroy(event_));
#endif
    }
  }

 protected:
  phi::Allocation *AllocateImpl(size_t size) override {
    PADDLE_ENFORCE_NOT_NULL(
        default_stream_,
        common::errors::PreconditionNotMet(
            "Default stream is not set for GPUContextAllocator"));
    platform::CUDADeviceGuard guard(place_.device);
    auto allocation = new GPUContextAllocation(
        static_unique_ptr_cast<Allocation>(memory::Alloc(place_, size)));
// Wait for the event on stream
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(event_, default_stream_));
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamWaitEvent(default_stream_, event_, 0));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event_, default_stream_));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(default_stream_, event_, 0));
#endif
    return allocation;
  }

  void FreeImpl(phi::Allocation *allocation) override { delete allocation; }

 private:
  phi::GPUPlace place_;
  gpuEvent_t event_{nullptr};
  gpuStream_t default_stream_{nullptr};
};

/**
 * GPUContextAllocatorPool is a singleton stores mapping from
 * CUDAPlace(s) to std::shared_ptr<GPUContextAllocator>. When a
 * phi::GPUContext's compute stream isn't default stream, it can call this
 * class to allocate GPU memory which will be released by a callback after
 * stream execution.
 */
class GPUContextAllocatorPool {
 public:
  static GPUContextAllocatorPool &Instance() {
    static GPUContextAllocatorPool pool;
    return pool;
  }

  AllocationPtr Alloc(const phi::GPUContext &dev_ctx, size_t size) {
    auto iter =
        allocators_.find(phi::GPUPlace(dev_ctx.GetPlace().GetDeviceId()));
    PADDLE_ENFORCE_NE(
        iter,
        allocators_.end(),
        common::errors::NotFound("No allocator found for CUDAPlace."));
    auto &allocator = iter->second;
    AllocationPtr allocation = allocator->Allocate(size);
    static_cast<GPUContextAllocation *>(allocation.get())
        ->SetGPUContext(&dev_ctx);
    return allocation;
  }

 private:
  GPUContextAllocatorPool() {
    std::vector<int> devices = platform::GetSelectedDevices();
    for (int i : devices) {
      auto place = phi::GPUPlace(i);
      auto compute_stream =
          phi::DeviceContextPool::Instance().GetByPlace(place)->stream();
      auto allocator = std::shared_ptr<GPUContextAllocator>(
          new GPUContextAllocator(place, compute_stream));
      allocators_.insert(make_pair(place, allocator));
    }
  }

  std::map<phi::GPUPlace, std::shared_ptr<GPUContextAllocator>> allocators_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
