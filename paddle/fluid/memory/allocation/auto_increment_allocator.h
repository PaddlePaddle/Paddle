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
class AutoIncrementAllocator : public ManagedAllocator {
 public:
  // Creator is the method to create ManagedAllocator
  using AllocatorCreator = std::function<std::shared_ptr<ManagedAllocator>()>;

  explicit AutoIncrementAllocator(AllocatorCreator&& creator)
      : creator_(std::move(creator)), prev_success_allocator_{0} {}
  std::unique_ptr<Allocation> Allocate(size_t size, Attr attr) override;
  std::shared_ptr<Allocation> AllocateShared(size_t size, Attr attr) override;
  bool IsAllocThreadSafe() const override;

 private:
  // NOTE: here use template Callback, it can be inlined when -O3
  template <typename Callback>
  inline typename std::result_of<Callback(ManagedAllocator&)>::type
  InvokeOrCreateUnderlyingAllocator(Callback callback) {
    std::shared_ptr<std::vector<AllocatorCreator::result_type>>
        underlying_allocators = underlying_allocators_;
    size_t retry_count = underlying_allocators->size();
    size_t allocator_num = retry_count;
    auto cur = prev_success_allocator_.load();
    while (retry_count-- > 0) {  // until there retry count is zero
      try {
        auto res = callback(*((*underlying_allocators)[cur]));
        prev_success_allocator_.store(cur);
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

    ManagedAllocator* new_allocator;
    {
      std::lock_guard<std::mutex> guard(mtx_);
      auto old_size = underlying_allocators_->size();
      decltype(underlying_allocators_) new_allocators(
          new std::vector<AllocatorCreator::result_type>(old_size + 1));
      for (size_t i = 0; i < old_size; ++i) {
        (*new_allocators)[i] = (*underlying_allocators_)[i];
      }

      (*new_allocators)[old_size] = creator_();
      new_allocator = (*new_allocators)[old_size].get();
      underlying_allocators_ = new_allocators;
      prev_success_allocator_.store(old_size);
    }

    PADDLE_ENFORCE(
        new_allocator->IsAllocThreadSafe(),
        "the underlying allocator must be thread safe. This is a program "
        "bug.");
    return callback(*new_allocator);
  }

  AllocatorCreator creator_;

  // Use std::shared_ptr to ensure thread-safety
  std::shared_ptr<std::vector<AllocatorCreator::result_type>>
      underlying_allocators_;

  // Use std::atomic rather than std::mutex, since std::atomic is usually
  // lock-free
  std::atomic<size_t> prev_success_allocator_{0};

  std::mutex mtx_;
};
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
