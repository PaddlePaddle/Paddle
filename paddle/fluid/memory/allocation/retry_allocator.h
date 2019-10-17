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

#include <atomic>              // NOLINT
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <utility>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

class RetryAllocator : public Allocator {
 public:
  RetryAllocator(std::shared_ptr<Allocator> allocator, size_t retry_ms)
      : underlying_allocator_(std::move(allocator)), retry_time_(retry_ms) {
    PADDLE_ENFORCE_NOT_NULL(
        underlying_allocator_,
        "UnderlyingAllocator of RetryAllocator must not be null");
    PADDLE_ENFORCE(underlying_allocator_->IsAllocThreadSafe(),
                   "UnderlyingAllocator of RetryAllocator must be thread-safe");
  }

  bool IsAllocThreadSafe() const override { return true; }

 protected:
  void FreeImpl(Allocation* allocation) override;
  Allocation* AllocateImpl(size_t size) override;

 private:
  std::shared_ptr<Allocator> underlying_allocator_;
  std::chrono::milliseconds retry_time_;
  std::mutex mutex_;
  std::condition_variable cv_;

  std::atomic<size_t> waited_allocate_size_{0};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
