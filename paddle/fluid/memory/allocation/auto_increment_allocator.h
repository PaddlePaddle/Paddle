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

#include <atomic>  // NOLINT
#include <functional>
#include <memory>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT
#include <vector>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

// The AutoIncrementAllocator manages many underlying allocators. If none of
// them can allocate the request memory, a new allocator will be created and
// invoke its `allocate` method.
//
// NOTE(yy): The AutoIncrementAllocator will prefer to allocate memory from
// the latest sucessful allocator.
//
// NOTE(yy): We may need to release an underlying allocator if it allocate
// nothing. However, it is generally not useful, since it will make performance
// undetermined.
//
// NOTE(yy): This allocator is only locked when creating new underlying
// allocator. The allocation requests from many threads may be dispatched
// to the same underlying allocator. So the underlying allocator must be
// thread safe.
//
// NOTE(zjl): Add capacity parameters to constructor. A high-performance
// thread-safe std::vector with varying size is hard to implement.
// Fortunately, we can get the total GPU memory and each chunk size.
// Therefore, we can get the suitable capacity of AutoIncrementAllocator.
class AutoIncrementAllocator : public ManagedAllocator {
 public:
  // Creator is the method to create ManagedAllocator
  using AllocatorCreator = std::function<std::shared_ptr<ManagedAllocator>()>;

  explicit AutoIncrementAllocator(AllocatorCreator&& creator, size_t capacity)
      : creator_(std::move(creator)), underlying_allocators_(capacity) {}
  std::unique_ptr<Allocation> Allocate(size_t size, Attr attr) override;
  std::shared_ptr<Allocation> AllocateShared(size_t size, Attr attr) override;
  bool IsAllocThreadSafe() const override;

 private:
  // NOTE: here use template Callback, it can be inlined when -O3
  template <typename Callback>
  inline typename std::result_of<Callback(ManagedAllocator&)>::type
  InvokeOrCreateUnderlyingAllocator(Callback callback) {
    auto cur = prev_success_allocator_.load();
    size_t retry_count = allocator_num_.load();
    size_t allocator_num = retry_count;
    while (retry_count-- > 0) {  // until there retry count is zero
      try {
        auto res = callback(*underlying_allocators_[cur]);
        prev_success_allocator_ = cur;
        return std::move(res);
      } catch (BadAlloc&) {
        if (++cur >= allocator_num) {
          cur = 0;
        }
      } catch (...) {
        // if there is another type of allocation, just rethrow it.
        std::rethrow_exception(std::current_exception());
      }
    }
    // No suitable allocator

    // This happens when the first allocator is exhausted and
    // there are more than 1 allocation requests
    // In this situation, the first allocation request would success
    // and the second allocation request would fail if we do not use
    // the newly created allocator by the first allocation request.
    for (size_t new_allocator_num = allocator_num_.load();
         allocator_num < new_allocator_num; ++allocator_num) {
      try {
        auto ret = callback(*underlying_allocators_[allocator_num]);
        prev_success_allocator_ = allocator_num;
        return std::move(ret);
      } catch (BadAlloc&) {
      } catch (...) {
        std::rethrow_exception(std::current_exception());
      }
    }

    ManagedAllocator* new_allocator;
    {
      std::lock_guard<std::mutex> guard(mtx_);
      auto old_size = allocator_num_.load();
      PADDLE_ENFORCE_LT(old_size, underlying_allocators_.size(),
                        "Allocator number exceeds capacity %d",
                        underlying_allocators_.size());
      underlying_allocators_[old_size] = creator_();
      new_allocator = underlying_allocators_[old_size].get();
      prev_success_allocator_ = old_size;
      allocator_num_.fetch_add(1);
    }

    PADDLE_ENFORCE(
        new_allocator->IsAllocThreadSafe(),
        "the underlying allocator must be thread safe. This is a program "
        "bug.");
    return callback(*new_allocator);
  }

  AllocatorCreator creator_;

  std::vector<AllocatorCreator::result_type> underlying_allocators_;
  std::atomic<size_t> allocator_num_{0};

  // Use std::atomic rather than std::mutex, since std::atomic is usually
  // lock-free
  std::atomic<size_t> prev_success_allocator_{0};

  std::mutex mtx_;
};
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
