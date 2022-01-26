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

#include "glog/logging.h"

namespace paddle {
namespace memory {
namespace allocation {

class WaitedAllocateSizeGuard {
 public:
  WaitedAllocateSizeGuard(std::atomic<size_t>* waited_size,
                          size_t requested_size)
      : waited_size_(waited_size), requested_size_(requested_size) {
    waited_size_->fetch_add(requested_size_,
                            std::memory_order::memory_order_relaxed);
  }

  ~WaitedAllocateSizeGuard() {
    waited_size_->fetch_sub(requested_size_,
                            std::memory_order::memory_order_relaxed);
  }

 private:
  std::atomic<size_t>* waited_size_;
  size_t requested_size_;
};

void RetryAllocator::FreeImpl(pten::Allocation* allocation) {
  // Delete underlying allocation first.
  size_t size = allocation->size();
  underlying_allocator_->Free(allocation);
  if (UNLIKELY(waited_allocate_size_)) {
    VLOG(10) << "Free " << size << " bytes and notify all waited threads, "
                                   "where waited_allocate_size_ = "
             << waited_allocate_size_;
    cv_.notify_all();
  }
}

pten::Allocation* RetryAllocator::AllocateImpl(size_t size) {
  auto alloc_func = [&, this]() {
    return underlying_allocator_->Allocate(size).release();
  };
  // In fact, we can unify the code of allocation success and failure
  // But it would add lock even when allocation success at the first time
  try {
    return alloc_func();
  } catch (BadAlloc&) {
    {
      WaitedAllocateSizeGuard guard(&waited_allocate_size_, size);
      VLOG(10) << "Allocation failed when allocating " << size
               << " bytes, waited_allocate_size_ = " << waited_allocate_size_;
      // We can just write allocation retry inside the predicate function of
      // wait_until. But it needs to acquire the lock when executing predicate
      // function. For better performance, we use loop here
      auto end_time = std::chrono::high_resolution_clock::now() + retry_time_;
      auto wait_until = [&, this] {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_until(lock, end_time);
      };

      size_t retry_time = 0;
      while (wait_until() != std::cv_status::timeout) {
        try {
          return alloc_func();
        } catch (BadAlloc&) {
          // do nothing when it is not timeout
          ++retry_time;
          VLOG(10) << "Allocation failed when retrying " << retry_time
                   << " times when allocating " << size
                   << " bytes. Wait still.";
        } catch (...) {
          throw;
        }
      }
    }
    VLOG(10) << "Allocation failed because of timeout when allocating " << size
             << " bytes.";
    return alloc_func();  // If timeout, try last allocation request.
  } catch (...) {
    throw;
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
