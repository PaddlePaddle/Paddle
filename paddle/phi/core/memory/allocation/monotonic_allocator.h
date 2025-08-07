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

#pragma once

#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <utility>

#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"

namespace paddle {
namespace memory {
namespace allocation {

class MonotonicAllocation;
class MonotonicAllocator : public Allocator {
 public:
  explicit MonotonicAllocator(std::shared_ptr<Allocator> underlying_allocator,
                              size_t alignment)
      : underlying_allocator_(std::move(underlying_allocator)),
        alignment_(alignment) {}

  bool IsAllocThreadSafe() const override { return true; }

  void AllocateBuffer(size_t capacity);
  void DeallocateBuffer();
  void ResetBuffer();

 protected:
  phi::Allocation* AllocateImpl(size_t unaligned_size) override;

  void FreeImpl(phi::Allocation* allocation) override;

  uint64_t ReleaseImpl(const phi::Place& place) override {
    return underlying_allocator_->Release(place);
  }

 protected:
  std::shared_ptr<Allocator> underlying_allocator_;
  AllocationPtr buffer_allocation_;

  size_t capacity_{0};
  size_t buffer_offset_{0};
  void* buffer_ptr_{nullptr};

  size_t alignment_{0};

  SpinLock spinlock_;
};

class MonotonicAllocation : public Allocation {
 public:
  explicit MonotonicAllocation(void* ptr, size_t size, phi::Place place)
      : Allocation(ptr, size, place) {}

  explicit MonotonicAllocation(DecoratedAllocationPtr underlying_allocation)
      : Allocation(underlying_allocation->ptr(),
                   underlying_allocation->base_ptr(),
                   underlying_allocation->size(),
                   underlying_allocation->place()),
        underlying_allocation_(std::move(underlying_allocation)) {}

 private:
  DecoratedAllocationPtr underlying_allocation_{nullptr};
};

class MonotonicAllocatorManager {
 public:
  using AllocatorMap = std::map<phi::GPUPlace, std::shared_ptr<Allocator>>;

  static MonotonicAllocatorManager& Instance() noexcept {
    static MonotonicAllocatorManager instance;
    return instance;
  }

  MonotonicAllocatorManager(const MonotonicAllocatorManager&) = delete;
  MonotonicAllocatorManager& operator=(const MonotonicAllocatorManager&) =
      delete;
  MonotonicAllocatorManager(MonotonicAllocatorManager&&) = delete;
  MonotonicAllocatorManager& operator=(MonotonicAllocatorManager&&) = delete;

  void Enable() { enabled_ = true; }
  void Disable() { enabled_ = false; }
  bool IsEnabled() const { return enabled_; }

  void SetAllocator(std::shared_ptr<Allocator> allocator,
                    const phi::Place& place);
  std::shared_ptr<Allocator> GetAllocator(const phi::Place& place);

  void AllocateBuffer(const phi::Place& place, size_t capacity);
  void DeallocateBuffer(const phi::Place& place);
  void ResetBuffer(const phi::Place& place);

 private:
  MonotonicAllocatorManager() = default;

  AllocatorMap allocators_{};
  bool enabled_{false};

  SpinLock spinlock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
