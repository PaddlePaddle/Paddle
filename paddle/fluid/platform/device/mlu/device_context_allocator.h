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

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"
#include "paddle/fluid/platform/device/mlu/mlu_info.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {

namespace platform {
class MLUDeviceContext;
}  // namespace platform

namespace memory {
namespace allocation {

/**
 * MLUDeviceContextAllocation is a wrapper of the underbeneath allocation.
 * MLUDeviceContextAllocation adds a MLU stream callback for the underbeneath
 * allocation so that MLUDeviceContextAllocation can be used in a MLU stream
 * which deletes allocation in the callback.
 */
class MLUDeviceContextAllocation : public Allocation {
 public:
  explicit MLUDeviceContextAllocation(AllocationPtr allocation)
      : Allocation(allocation->ptr(), allocation->size(), allocation->place()),
        underlying_allocation_(std::move(allocation)) {}

  ~MLUDeviceContextAllocation() {
    PADDLE_ENFORCE_NOT_NULL(
        dev_ctx_,
        platform::errors::PreconditionNotMet(
            "Device context is not set for MLUDeviceContextAllocation"));
    auto *p_allocation = underlying_allocation_.release();
    VLOG(4) << "Adding callback to delete MLUDeviceContextAllocation at "
            << p_allocation;
    dev_ctx_->AddStreamCallback([p_allocation] {
      VLOG(4) << "Delete MLUDeviceContextAllocation at " << p_allocation;
      AllocationDeleter()(p_allocation);
    });
  }

  void SetMLUDeviceContext(const platform::MLUDeviceContext *dev_ctx) {
    dev_ctx_ = dev_ctx;
  }

 private:
  AllocationPtr underlying_allocation_;
  const platform::MLUDeviceContext *dev_ctx_{nullptr};
};

/**
 * MLUDeviceContextAllocator will allocate a MLUDeviceContextAllocation
 * after waiting for a self-created event on the default stream. It does so to
 * let the non-default stream be able to allocate GPU memory which will be
 * released by stream callback
 */
class MLUDeviceContextAllocator : public Allocator {
 public:
  explicit MLUDeviceContextAllocator(platform::MLUPlace place,
                                     mluStream default_stream)
      : place_(place), default_stream_(default_stream) {
    platform::MLUDeviceGuard guard(place_.device);
    PADDLE_ENFORCE_MLU_SUCCESS(cnrtNotifierCreate(&event_));
  }

  ~MLUDeviceContextAllocator() {
    if (event_) {
      platform::MLUDeviceGuard guard(place_.device);
      PADDLE_ENFORCE_MLU_SUCCESS(cnrtNotifierDestroy(event_));
    }
  }

 protected:
  Allocation *AllocateImpl(size_t size) override {
    PADDLE_ENFORCE_NOT_NULL(
        default_stream_,
        platform::errors::PreconditionNotMet(
            "Default stream is not set for MLUDeviceContextAllocator"));
    platform::MLUDeviceGuard guard(place_.device);
    auto allocation =
        new MLUDeviceContextAllocation(memory::Alloc(place_, size));
    // Wait for the event on stream
    PADDLE_ENFORCE_MLU_SUCCESS(cnrtPlaceNotifier(event_, default_stream_));
    PADDLE_ENFORCE_MLU_SUCCESS(cnrtWaitNotifier(event_));
    return allocation;
  }

  void FreeImpl(Allocation *allocation) override { delete allocation; }

 private:
  platform::MLUPlace place_;
  mluEventHandle event_{nullptr};
  mluStream default_stream_{nullptr};
};

/**
 * MLUDeviceContextAllocatorPool is a singletion stores mapping from
 * MLUPlace(s) to std::shared_ptr<MLUDeviceContextAllocator>. When a
 * MLUDeviceContext's compute stream isn't default stream, it can call this
 * class to allocate GPU memory which will be released by a callback after
 * stream execution.
 */
class MLUDeviceContextAllocatorPool {
 public:
  static MLUDeviceContextAllocatorPool &Instance() {
    static MLUDeviceContextAllocatorPool pool;
    return pool;
  }

  AllocationPtr Alloc(const platform::MLUDeviceContext &dev_ctx, size_t size) {
    auto iter = allocators_.find(
        BOOST_GET_CONST(platform::MLUPlace, dev_ctx.GetPlace()));
    PADDLE_ENFORCE_NE(
        iter, allocators_.end(),
        platform::errors::NotFound("No allocator found for MLUPlace."));
    auto &allocator = iter->second;
    AllocationPtr allocation = allocator->Allocate(size);
    static_cast<MLUDeviceContextAllocation *>(allocation.get())
        ->SetMLUDeviceContext(&dev_ctx);
    return allocation;
  }

 private:
  MLUDeviceContextAllocatorPool() {
    std::vector<int> devices = platform::GetMLUSelectedDevices();
    for (int i : devices) {
      auto place = platform::MLUPlace(i);
      auto compute_stream =
          platform::DeviceContextPool::Instance().GetByPlace(place)->stream();
      auto allocator = std::shared_ptr<MLUDeviceContextAllocator>(
          new MLUDeviceContextAllocator(place, compute_stream));
      allocators_.insert(make_pair(place, allocator));
    }
  }

  std::map<platform::MLUPlace, std::shared_ptr<MLUDeviceContextAllocator>>
      allocators_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
