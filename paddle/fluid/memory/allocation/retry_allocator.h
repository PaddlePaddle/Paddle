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

#pragma once

#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

class RetryAllocator;

class RetryAllocation : public Allocation {
 public:
  RetryAllocation(std::unique_ptr<Allocation>&& underlying_allocation,
                  const std::shared_ptr<RetryAllocator>& retry_allocator)
      : Allocation(underlying_allocation->ptr(), underlying_allocation->size(),
                   underlying_allocation->place()),
        underlying_allocation_(std::move(underlying_allocation)),
        retry_allocator_(retry_allocator) {}

  ~RetryAllocation() final;

 private:
  std::unique_ptr<Allocation> underlying_allocation_;
  std::weak_ptr<RetryAllocator> retry_allocator_;
};

class RetryAllocator : public ManagedAllocator,
                       public std::enable_shared_from_this<RetryAllocator> {
 private:
  RetryAllocator(std::unique_ptr<Allocator>&& allocator, size_t retry_ms)
      : underlying_allocator_(
            dynamic_cast<UnmanagedAllocator*>(allocator.release())),
        retry_time_(retry_ms) {
    EnforceCheck();
  }

 public:
  template <typename... Args>
  static std::shared_ptr<ManagedAllocator> Create(Args... args) {
    return std::shared_ptr<ManagedAllocator>(
        new RetryAllocator(std::forward<Args>(args)...));
  }

  bool IsAllocThreadSafe() const override;

  std::unique_ptr<Allocation> Allocate(size_t size,
                                       Allocator::Attr attr) override;

  std::shared_ptr<Allocation> AllocateShared(size_t size,
                                             Allocator::Attr attr) override;

  void FreeUnderlyingAllocation(std::unique_ptr<Allocation>&& allocation);

 private:
  Allocation* AllocateImpl(size_t size, Allocator::Attr attr);

  void EnforceCheck() {
    PADDLE_ENFORCE_NOT_NULL(
        underlying_allocator_.get(),
        "UnderlyingAllocator of RetryAllocator must be UnmanagedAllocator");
    PADDLE_ENFORCE(underlying_allocator_->IsAllocThreadSafe(),
                   "UnderlyingAllocator of RetryAllocator must be thread-safe");
  }

  std::unique_ptr<UnmanagedAllocator> underlying_allocator_;
  std::chrono::milliseconds retry_time_;
  std::mutex mutex_;
  std::condition_variable cv_;

  // For debug, We can add an atomic integer to record how many memory sizes are
  // waited to allocate
  // std::atomic<size_t> waited_allocate_size_{0};

  friend class RetryAllocation;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
