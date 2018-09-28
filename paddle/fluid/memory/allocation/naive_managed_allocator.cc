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

#include "paddle/fluid/memory/allocation/naive_managed_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

NaiveManagedAllocator::NaiveManagedAllocator(
    std::unique_ptr<Allocator> &&allocator) {
  auto *underlying_allocator =
      dynamic_cast<UnmanagedAllocator *>(allocator.get());
  PADDLE_ENFORCE_NOT_NULL(underlying_allocator);
  allocator.release();
  Init(std::unique_ptr<UnmanagedAllocator>(underlying_allocator));
}

NaiveManagedAllocator::NaiveManagedAllocator(
    std::unique_ptr<UnmanagedAllocator> &&allocator) {
  Init(std::move(allocator));
}
void NaiveManagedAllocator::Init(
    std::unique_ptr<UnmanagedAllocator> &&allocator) {
  underlying_allocator_ = std::move(allocator);
}
bool NaiveManagedAllocator::IsAllocThreadSafe() const {
  return underlying_allocator_->IsAllocThreadSafe();
}
std::unique_ptr<Allocation> NaiveManagedAllocator::Allocate(size_t size,
                                                            Attr attr) {
  std::unique_ptr<Allocation> allocation =
      underlying_allocator_->Allocate(size, attr);
  return std::unique_ptr<Allocation>(
      new NaiveManagedAllocation(std::move(allocation), shared_from_this()));
}
std::shared_ptr<Allocation> NaiveManagedAllocator::AllocateShared(size_t size,
                                                                  Attr attr) {
  std::unique_ptr<Allocation> allocation =
      underlying_allocator_->Allocate(size, attr);
  return std::shared_ptr<Allocation>(
      new NaiveManagedAllocation(std::move(allocation), shared_from_this()));
}

NaiveManagedAllocation::~NaiveManagedAllocation() {
  auto allocator = allocator_.lock();
  if (UNLIKELY(allocator == nullptr)) {
    // the allocator is destructed before allocations.
    // do nothing.
    return;
  }
  // invoke Free
  allocator->UnderlyingAllocator().FreeUniquePtr(
      std::move(underlying_allocation_));
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
