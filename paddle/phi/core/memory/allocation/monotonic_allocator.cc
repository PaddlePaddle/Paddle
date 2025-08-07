// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/monotonic_allocator.h"

#include <algorithm>
#include <mutex>  // NOLINT
#include <utility>

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif

namespace paddle::memory::allocation {

void MonotonicAllocator::AllocateBuffer(size_t capacity) {
  std::lock_guard<SpinLock> guard(spinlock_);
  if (buffer_ptr_ == nullptr) {
    buffer_allocation_ = std::move(underlying_allocator_->Allocate(capacity));
    buffer_ptr_ = buffer_allocation_->ptr();
    buffer_offset_ = 0;
    capacity_ = capacity;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "MonotonicAllocator::Allocate: buffer_ptr_ is not null"));
  }
}

void MonotonicAllocator::DeallocateBuffer() {
  std::lock_guard<SpinLock> guard(spinlock_);
  buffer_offset_ = 0;
  buffer_ptr_ = nullptr;
  capacity_ = 0;
  buffer_allocation_.reset();
}

void MonotonicAllocator::ResetBuffer() {
  std::lock_guard<SpinLock> guard(spinlock_);
  buffer_offset_ = 0;
}

phi::Allocation* MonotonicAllocator::AllocateImpl(size_t unaligned_size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t size = AlignedSize(unaligned_size, alignment_);

  size_t free_size = capacity_ - buffer_offset_;

  VLOG(10) << "Allocate " << unaligned_size << " bytes, aligned to " << size
           << ", Free size " << free_size;

  if (free_size >= size) {
    if (buffer_allocation_ == nullptr) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "MonotonicAllocator::Allocate: buffer_allocation_ is null"));
    }
    void* ptr = reinterpret_cast<char*>(buffer_ptr_) + buffer_offset_;
    buffer_offset_ += size;
    return new MonotonicAllocation(ptr, size, buffer_allocation_->place());
  } else {
    VLOG(10) << "Allocate fail, fallback to underlying_allocator";
    AllocationPtr underlying_allocation = underlying_allocator_->Allocate(size);
    // Leveraging the features of unique_ptr, the Deleter can be automatically
    // invoked in FreeImpl to release the underlying_allocation.
    return new MonotonicAllocation(
        static_unique_ptr_cast<Allocation>(std::move(underlying_allocation)));
  }
}

void MonotonicAllocator::FreeImpl(phi::Allocation* allocation) {
  VLOG(10) << "Free " << allocation->size()
           << " bytes, ptr = " << allocation->ptr();
  std::lock_guard<SpinLock> guard(spinlock_);
  delete allocation;
}

std::shared_ptr<Allocator> MonotonicAllocatorManager::CreateOrGetAllocator(
    const phi::Place& place, phi::Allocator* underlying_allocator) {
  std::lock_guard<SpinLock> guard(spinlock_);
#ifdef PADDLE_WITH_CUDA
  size_t alignment = platform::GpuMinChunkSize();
#else
  size_t alignment = 1 << 8;
#endif
  if (allocators_.find(place) == allocators_.end()) {
    allocators_[place] =
        std::make_shared<MonotonicAllocator>(underlying_allocator, alignment);
  }
  return allocators_[place];
}

void MonotonicAllocatorManager::AllocateBuffer(const phi::Place& place,
                                               size_t capacity) {
  std::lock_guard<SpinLock> guard(spinlock_);

  if (allocators_.find(place) != allocators_.end()) {
    auto monotonic_allocator =
        std::dynamic_pointer_cast<MonotonicAllocator>(allocators_[place]);
    if (monotonic_allocator) {
      monotonic_allocator->AllocateBuffer(capacity);
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "MonotonicAllocatorManager: place %s is not initialized.",
        place.DebugString()));
  }
}

void MonotonicAllocatorManager::DeallocateBuffer(const phi::Place& place) {
  std::lock_guard<SpinLock> guard(spinlock_);
  if (allocators_.find(place) != allocators_.end()) {
    auto monotonic_allocator =
        std::dynamic_pointer_cast<MonotonicAllocator>(allocators_[place]);
    if (monotonic_allocator) {
      monotonic_allocator->DeallocateBuffer();
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "MonotonicAllocatorManager: place %s is not initialized.",
        place.DebugString()));
  }
}

void MonotonicAllocatorManager::ResetBuffer(const phi::Place& place) {
  std::lock_guard<SpinLock> guard(spinlock_);
  if (allocators_.find(place) != allocators_.end()) {
    auto monotonic_allocator =
        std::dynamic_pointer_cast<MonotonicAllocator>(allocators_[place]);
    if (monotonic_allocator) {
      monotonic_allocator->ResetBuffer();
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "MonotonicAllocatorManager: place %s is not initialized.",
        place.DebugString()));
  }
}

}  // namespace paddle::memory::allocation
