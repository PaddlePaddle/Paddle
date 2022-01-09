/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstdint>
#include "paddle/fluid/platform/place.h"

namespace pten {
namespace candidate {

/// \brief Fancy pointer with deleter. The use of this data type
/// is to be compatible with allocators from different frameworks
/// without significant performance loss. This class does not
/// support being inherited.
class Allocation {
 public:
  using Place = paddle::platform::Place;
  using DeleterFnPtr = void (*)(Allocation*);

  Allocation() = default;

  // Don't own resources, only provide access.
  Allocation(void* data, size_t size, const Place& place)
      : ptr_(data), size_(size), place_(place) {}

  // Own resources.
  Allocation(void* data, size_t size, DeleterFnPtr deleter, const Place& place)
      : ptr_(data), size_(size), deleter_(deleter), place_(place) {}

  Allocation(Allocation&& other) noexcept {
    swap(*this, other);
    CHECK(other.deleter_ == nullptr);
  }
  Allocation& operator=(Allocation&& other) noexcept {
    // Exchange them explicitly to avoid moving is equivalent
    // to copying.
    swap(*this, other);
    CHECK(other.deleter_ == nullptr);
    return *this;
  }

  virtual ~Allocation() {
    if (deleter_) {
      deleter_(this);
    }
  }

  // Returns the holding pointer.
  // NOTE: For performance consideration, it is better not to make this method
  // as a virtual method. If we want to implement a `defragmentation` later,
  // we might need to make `ptr_` field as a protected field, and add a virtual
  // method like `defragmentation` to change `ptr_`.
  void* ptr() const noexcept { return ptr_; }

  // Returns the size of this memory buffer, i.e., ptr() + size() - 1 is the
  // last valid element.
  //
  // NOTE: Some allocator might alloc more memory than request. The size
  // could larger than its request. For example,
  //    the AlignedAllocator will always allocate memory as size + kAlignment.
  //    The raw pointer might not aligned, so an offset might be added to raw
  //    the pointer. The size of this allocation will be
  //    `size + kAlignemnt - offset`.
  size_t size() const noexcept { return size_; }

  void* operator->() const noexcept { return ptr_; }
  operator bool() const noexcept { return ptr_; }
  const Place& place() const noexcept { return place_; }
  DeleterFnPtr deleter() const noexcept { return deleter_; }

 protected:
  friend void swap(Allocation& a, Allocation& b) noexcept;
  void* ptr_{nullptr};
  size_t size_{};
  DeleterFnPtr deleter_{nullptr};
  // TODO(Shixiaowei02): Enum needs to be used instead to reduce
  // the construction overhead by more than 50%.
  Place place_;
};

inline void swap(Allocation& a, Allocation& b) noexcept {
  ::std::swap(a.ptr_, b.ptr_);
  ::std::swap(a.deleter_, b.deleter_);
  ::std::swap(a.place_, b.place_);
  ::std::swap(a.size_, b.size_);
}

class AllocationDeleter {
 public:
  using DeleteFnPtr = void (*)(Allocation*);
  AllocationDeleter() = default;
  AllocationDeleter(DeleteFnPtr deleter) : deleter_(deleter) {}  // NOLINT

  void operator()(Allocation* allocation) const {
    if (deleter_) {
      deleter_(allocation);
    } else {
      delete allocation;
    }
  }

 private:
  DeleteFnPtr deleter_{nullptr};
};

using AllocationPtr = std::unique_ptr<Allocation, AllocationDeleter>;

class Allocator {
 public:
  using Place = paddle::platform::Place;

  virtual ~Allocator() = default;
  virtual AllocationPtr Allocate(size_t bytes_size) = 0;
  virtual void Free(Allocation* allocation) = 0;

  virtual void* AllocateRaw(size_t bytes_size) { return nullptr; }
  virtual void DeallocateRaw(void* ptr, size_t bytes_size) {}

  virtual bool IsAllocThreadSafe() const { return false; }
};

}  // namespace candidate
}  // namespace pten
