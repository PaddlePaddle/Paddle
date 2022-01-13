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

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {

namespace platform {
class CUDADeviceContext;
}  // namespace platform

namespace memory {
namespace allocation {

/**
 * CUDADeviceContextAllocation is a wrapper of the underbeneath allocation.
 * CUDADeviceContextAllocation adds a CUDA stream callback for the underbeneath
 * allocation so that CUDADeviceContextAllocation can be used in a CUDA stream
 * which deletes allocation in the callback.
 */
class CUDADeviceContextAllocation : public Allocation {
 public:
  explicit CUDADeviceContextAllocation(DecoratedAllocationPtr allocation)
      : Allocation(allocation->ptr(), allocation->base_ptr(),
                   allocation->size(), allocation->place()),
        underlying_allocation_(std::move(allocation)) {}

  ~CUDADeviceContextAllocation() {
    PADDLE_ENFORCE_NOT_NULL(
        dev_ctx_,
        platform::errors::PreconditionNotMet(
            "Device context is not set for CUDADeviceContextAllocation"));
    auto *p_allocation = underlying_allocation_.release();
    VLOG(4) << "Adding callback to delete CUDADeviceContextAllocation at "
            << p_allocation;
    dev_ctx_->AddStreamCallback([p_allocation] {
      VLOG(4) << "Delete CUDADeviceContextAllocation at " << p_allocation;
      Allocator::AllocationDeleter(p_allocation);
    });
  }

  void SetCUDADeviceContext(const platform::CUDADeviceContext *dev_ctx) {
    dev_ctx_ = dev_ctx;
  }

 private:
  DecoratedAllocationPtr underlying_allocation_;
  const platform::CUDADeviceContext *dev_ctx_{nullptr};
};

/**
 * CUDADeviceContextAllocator will allocate a CUDADeviceContextAllocation
 * after waiting for a self-created event on the default stream. It does so to
 * let the non-default stream be able to allocate GPU memory which will be
 * released by stream callback
 */
class CUDADeviceContextAllocator : public Allocator {
 public:
  explicit CUDADeviceContextAllocator(platform::CUDAPlace place,
                                      gpuStream_t default_stream)
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

  ~CUDADeviceContextAllocator() {
    if (event_) {
      platform::CUDADeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(event_));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event_));
#endif
    }
  }

 protected:
  pten::Allocation *AllocateImpl(size_t size) override {
    PADDLE_ENFORCE_NOT_NULL(
        default_stream_,
        platform::errors::PreconditionNotMet(
            "Default stream is not set for CUDADeviceContextAllocator"));
    platform::CUDADeviceGuard guard(place_.device);
    auto allocation = new CUDADeviceContextAllocation(
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

  void FreeImpl(pten::Allocation *allocation) override { delete allocation; }

 private:
  platform::CUDAPlace place_;
  gpuEvent_t event_{nullptr};
  gpuStream_t default_stream_{nullptr};
};

/**
 * CUDADeviceContextAllocatorPool is a singletion stores mapping from
 * CUDAPlace(s) to std::shared_ptr<CUDADeviceContextAllocator>. When a
 * CUDADeviceContext's compute stream isn't default stream, it can call this
 * class to allocate GPU memory which will be released by a callback after
 * stream execution.
 */
class CUDADeviceContextAllocatorPool {
 public:
  static CUDADeviceContextAllocatorPool &Instance() {
    static CUDADeviceContextAllocatorPool pool;
    return pool;
  }

  AllocationPtr Alloc(const platform::CUDADeviceContext &dev_ctx, size_t size) {
    auto iter =
        allocators_.find(platform::CUDAPlace(dev_ctx.GetPlace().GetDeviceId()));
    PADDLE_ENFORCE_NE(
        iter, allocators_.end(),
        platform::errors::NotFound("No allocator found for CUDAPlace."));
    auto &allocator = iter->second;
    AllocationPtr allocation = allocator->Allocate(size);
    static_cast<CUDADeviceContextAllocation *>(allocation.get())
        ->SetCUDADeviceContext(&dev_ctx);
    return allocation;
  }

 private:
  CUDADeviceContextAllocatorPool() {
    std::vector<int> devices = platform::GetSelectedDevices();
    for (int i : devices) {
      auto place = platform::CUDAPlace(i);
      auto compute_stream =
          platform::DeviceContextPool::Instance().GetByPlace(place)->stream();
      auto allocator = std::shared_ptr<CUDADeviceContextAllocator>(
          new CUDADeviceContextAllocator(place, compute_stream));
      allocators_.insert(make_pair(place, allocator));
    }
  }

  std::map<platform::CUDAPlace, std::shared_ptr<CUDADeviceContextAllocator>>
      allocators_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
