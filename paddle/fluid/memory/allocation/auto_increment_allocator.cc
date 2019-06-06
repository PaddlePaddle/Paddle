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

#include "paddle/fluid/memory/allocation/auto_increment_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {
bool AutoIncrementAllocator::IsAllocThreadSafe() const { return true; }

std::shared_ptr<Allocator> AutoIncrementAllocator::CreateNewAllocator() {
  std::lock_guard<std::mutex> guard(mtx_);
  auto old_size = allocator_num_.load();
  PADDLE_ENFORCE_LT(old_size, underlying_allocators_.size(),
                    "Allocator number exceeds capacity %d",
                    underlying_allocators_.size());
  underlying_allocators_[old_size] = creator_();
  prev_success_allocator_ = old_size;
  ++allocator_num_;
  PADDLE_ENFORCE(
      underlying_allocators_[old_size]->IsAllocThreadSafe(),
      "the underlying allocator must be thread safe. This is a program "
      "bug.");
  return underlying_allocators_[old_size];
}
Allocation *AutoIncrementAllocator::AllocateImpl(size_t size) {
  auto cur = prev_success_allocator_.load();
  size_t retry_count = allocator_num_.load();
  size_t allocator_num = retry_count;
  while (retry_count-- > 0) {  // until there retry count is zero
    try {
      auto res = underlying_allocators_[cur]->Allocate(size);
      prev_success_allocator_ = cur;
      return res.release();
    } catch (BadAlloc &) {
      if (++cur >= allocator_num) {
        cur = 0;
      }
    } catch (...) {
      // if there is another type of allocation, just rethrow it.
      throw;
    }
  }

  // This happens when the first allocator is exhausted and
  // there are more than 1 allocation requests
  // In this situation, the first allocation request would success
  // and the second allocation request would fail if we do not use
  // the newly created allocator by the first allocation request.
  for (cur = allocator_num; cur < allocator_num_; ++cur) {
    try {
      auto ret = underlying_allocators_[cur]->Allocate(size);
      prev_success_allocator_ = cur;
      return ret.release();
    } catch (BadAlloc &) {
    } catch (...) {
      throw;
    }
  }
  // No suitable allocator
  return CreateNewAllocator()->Allocate(size).release();
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
