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
#include "paddle/fluid/memory/allocation/allocation_with_underlying.h"
namespace paddle {
namespace memory {
namespace allocation {

void RetryAllocator::FreeImpl(Allocation* allocation) {
  // Delete underlying allocation first.
  underlying_allocator_->Free(allocation);
  cv_.notify_all();
}

Allocation* RetryAllocator::AllocateImpl(size_t size, Allocator::Attr attr) {
  auto alloc_func = [&, this]() {
    return underlying_allocator_->Allocate(size, attr).release();
  };
  // In fact, we can unify the code of allocation success and failure
  // But it would add lock even when allocation success at the first time
  try {
    return alloc_func();
  } catch (BadAlloc& bad_alloc) {
    {
      // We can just write allocation retry inside the predicate function of
      // wait_until
      // But it needs to acquire the lock when executing predicate function
      // For better performance, we use loop here
      auto end_time = std::chrono::high_resolution_clock::now() + retry_time_;
      auto wait_until = [&, this] {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_until(lock, end_time);
      };
      while (wait_until() != std::cv_status::timeout) {
        try {
          return alloc_func();
        } catch (BadAlloc& ex) {
          bad_alloc = ex;
        } catch (...) {
          throw;
        }
      }

      throw;  // rethrow the original exception or throw the internal bad_alloc
    }
  } catch (...) {
    throw;
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
