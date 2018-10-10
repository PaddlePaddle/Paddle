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

#include "paddle/fluid/memory/allocation/retry_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

RetryAllocation::~RetryAllocation() {
  auto allocator = retry_allocator_.lock();
  {
    // release allocation first
    if (UNLIKELY(allocator == nullptr)) return;
    allocator->underlying_allocator_->Free(underlying_allocation_.release());
  }

  {
    // notify all waited allocators
    std::lock_guard<std::mutex> lock(allocator->mutex_);
    allocator->cv_.notify_all();
  }
}

bool RetryAllocator::IsAllocThreadSafe() const { return true; }

std::shared_ptr<Allocation> RetryAllocator::AllocateShared(
    size_t size, Allocator::Attr attr) {
  return std::shared_ptr<Allocation>(Allocate(size, attr));
}

std::unique_ptr<Allocation> RetryAllocator::Allocate(size_t size,
                                                     Allocator::Attr attr) {
  auto alloc_func = [&, this]() {
    return new RetryAllocation(underlying_allocator_->Allocate(size, attr),
                               this->shared_from_this());
  };

  // In fact, we can unify the code of allocation success and failure
  // But it would add lock even when allocation success at the first time
  std::unique_ptr<Allocation> ret;
  try {
    ret.reset(alloc_func());
  } catch (BadAlloc &) {
    {
      // We can just write allocation retry inside the predicate function of
      // wait_until
      // But it needs to acquire the lock when executing predicate function
      // For better performance, we use loop here
      std::exception_ptr ex;
      auto end_time = std::chrono::high_resolution_clock::now() + retry_time_;
      std::cv_status status;
      do {
        {
          std::unique_lock<std::mutex> lock(mutex_);
          status = cv_.wait_until(lock, end_time);
        }
        try {
          ret.reset(alloc_func());
        } catch (BadAlloc &) {
          ex = std::current_exception();
        } catch (...) {
          std::rethrow_exception(std::current_exception());
        }
      } while (ret == nullptr && status != std::cv_status::timeout);

      if (ret == nullptr) std::rethrow_exception(ex);
    }
  } catch (...) {
    std::rethrow_exception(std::current_exception());
  }
  return ret;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
