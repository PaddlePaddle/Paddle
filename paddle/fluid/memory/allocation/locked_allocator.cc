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

#include "paddle/fluid/memory/allocation/locked_allocator.h"
#include <mutex>  // NOLINT

namespace paddle {
namespace memory {
namespace allocation {

std::unique_ptr<Allocation> LockedAllocator::Allocate(size_t size, Attr attr) {
  if (underlying_allocator_->IsAllocThreadSafe()) {
    return underlying_allocator_->Allocate(size, attr);
  } else {
    std::lock_guard<std::mutex> guard(mtx_);
    return underlying_allocator_->Allocate(size, attr);
  }
}
void LockedAllocator::FreeUniquePtr(std::unique_ptr<Allocation> allocation) {
  if (underlying_allocator_->IsAllocThreadSafe()) {
    return underlying_allocator_->FreeUniquePtr(std::move(allocation));
  } else {
    std::lock_guard<std::mutex> guard(mtx_);
    return underlying_allocator_->FreeUniquePtr(std::move(allocation));
  }
}
bool LockedAllocator::IsAllocThreadSafe() const { return true; }

LockedAllocator::LockedAllocator(
    std::unique_ptr<Allocator> &&underlying_allocator) {
  auto *allocator =
      dynamic_cast<UnmanagedAllocator *>(underlying_allocator.get());
  PADDLE_ENFORCE_NOT_NULL(allocator);
  underlying_allocator.release();
  underlying_allocator_.reset(allocator);
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
