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

#include "paddle/fluid/memory/allocation/buffered_allocator.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace paddle {
namespace memory {
namespace allocation {

BufferedAllocator::BufferedAllocator(std::unique_ptr<Allocator>&& allocator) {
  underlying_allocator_.reset(
      dynamic_cast<UnmanagedAllocator*>(allocator.release()));
  PADDLE_ENFORCE_NOT_NULL(
      underlying_allocator_,
      "Underlying allocator of BufferedAllocator must be unmanaged");
  if (underlying_allocator_->IsAllocThreadSafe()) {
    mtx_.reset(new std::mutex());
  }
}

BufferedAllocator::~BufferedAllocator() { FreeCache(-1UL); }

std::unique_ptr<Allocation> BufferedAllocator::Allocate(size_t size,
                                                        Allocator::Attr attr) {
  std::unique_ptr<Allocation> result;
  {
    platform::LockGuardPtr<std::mutex> guard(mtx_);
    auto it = allocations_.lower_bound(size);
    if (it != allocations_.end() && it->first < size * 2) {
      result = std::move(it->second);
      allocations_.erase(it);
    }
  }

  if (result) {
    return result;
  }

  try {
    return underlying_allocator_->Allocate(size, attr);
  } catch (BadAlloc&) {
    FreeCache(size);
    return underlying_allocator_->Allocate(size, attr);
  }
}

void BufferedAllocator::FreeCache(size_t size) {
  platform::LockGuardPtr<std::mutex> guard(mtx_);
  if (UNLIKELY(size == 0)) return;
  size_t cur = 0;
  while (!allocations_.empty()) {  // free the largest
    auto it = --allocations_.end();
    cur += it->second->size();
    underlying_allocator_->FreeUniquePtr(std::move(it->second));
    allocations_.erase(it);
    if (cur >= size) return;
  }
}

void BufferedAllocator::FreeUniquePtr(std::unique_ptr<Allocation> allocation) {
  platform::LockGuardPtr<std::mutex> guard(mtx_);
  allocations_.emplace(allocation->size(), std::move(allocation));
}

bool BufferedAllocator::IsAllocThreadSafe() const {
  return this->underlying_allocator_->IsAllocThreadSafe();
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
