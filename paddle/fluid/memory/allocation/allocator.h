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
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/inlined_vector.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

// Exception when `Alloc`/`AllocShared` failed
struct BadAlloc : public std::exception {
  inline explicit BadAlloc(std::string err_msg, const char* file, int line)
      : err_str_(platform::GetTraceBackString(std::move(err_msg), file, line)) {
  }

  const char* what() const noexcept override { return err_str_.c_str(); }

  std::string err_str_;
};

class Allocator;

// Allocation is the object holding the actually pointer. Use
// `Allocation::ptr()` will returns the pointer that allocated.
//
// NOTE: this is the base class of Allocation. Each allocator can use its own
//       allocation object.
// NOTE: the `Allocation::ptr()` could be nullptr, if the allocation size is 0

/**
 * Allocation is returned by Allocator::Allocate() method.
 *
 * An allocator may be decorated by another allocator. For example, we can
 * decorate a RetryAllocator to any allocator to perform allocation retry when
 * first allocation request fails.
 *
 * Explanations of Allocator design are as follows:
 *
 * Suppose we have an allocator which is decorated by several allocators:
 *
 *   A(1) <- A(2) <- A(3) <- ... <- A(n)
 *
 * , and the public allocator is A(1).
 *
 * The allocation process would be:
 *
 *   A(n).Allocate() -> ... -> A(2).Allocate() -> A(1).Allocate()
 *
 * , and the free process would be:
 *
 *   A(1).Free() -> A(2).Free() -> ... -> A(n).Free()
 *
 * Therefore, we should record the allocator chain when allocating, so
 * that we can free the allocation in the reverse order of allocator chain.
 * The field `decorated_allocators_` is used to record this chain.
 *
 * Another example is that we want to add additional fields in Allocation,
 * e.g., something what is done in AlignedAllocator, etc.
 * In this case, we should declare a derived class of Allocation, which
 * contains an underlying Allocation allocated by the underlying allocator.
 * Therefore, `decorated_allocators_` of the new Allocation object would
 * be a new chain, differing from the underlying Allocation object.
 */
class Allocation {
 public:
  inline Allocation(void* ptr, size_t size, platform::Place place)
      : ptr_(ptr), size_(size), place_(place) {}

  Allocation(const Allocation& o) = delete;
  Allocation& operator=(const Allocation& o) = delete;
  Allocation(Allocation&& o) = delete;
  Allocation& operator=(Allocation&& o) = delete;

  // Returns the holding pointer.
  // NOTE: For performance consideration, it is better not to make this method
  // as a virtual method. If we want to implement a `defragmentation` later,
  // we might need to make `ptr_` field as a protected field, and add a virtual
  // method like `defragmentation` to change `ptr_`.
  inline void* ptr() const { return ptr_; }

  // Returns the size of this memory buffer, i.e., ptr() + size() - 1 is the
  // last valid element.
  //
  // NOTE: Some allocator might alloc more memory than request. The size
  // could larger than its request. For example,
  //    the AlignedAllocator will always allocate memory as size + kAlignment.
  //    The raw pointer might not aligned, so an offset might be added to raw
  //    the pointer. The size of this allocation will be
  //    `size + kAlignemnt - offset`.
  inline size_t size() const { return size_; }

  inline const platform::Place& place() const { return place_; }

  virtual ~Allocation() {}

 private:
  inline void RegisterDecoratedAllocator(Allocator* allocator) {
    decorated_allocators_.emplace_back(allocator);
  }

  inline void PopDecoratedAllocator() { decorated_allocators_.pop_back(); }

  inline Allocator* TopDecoratedAllocator() {
    return decorated_allocators_.back();
  }

 private:
  void* ptr_;
  size_t size_;
  platform::Place place_;

  /**
   * NOTE(zjl): Since decorated_allocators_ is usually a small vector.
   * We reserve a small buffer to it to prevent frequent heap allocation
   *
   * Instead, we can use a std::vector<Allocator *> here, and reserve
   * kReserveAllocatorNum in constructor of Allocation.
   * But using std::vector<Allocator *> would make ocr recognition model
   * fail in CE. The train duration is 8% slower than KPI.
   */
  static constexpr size_t kReserveAllocatorNum = 8;
  using DecoratedAllocatorStack =
      framework::InlinedVector<Allocator*, kReserveAllocatorNum>;

  DecoratedAllocatorStack decorated_allocators_;

  friend class Allocator;
};

// Base interface class of memory Allocator.
class Allocator {
 public:
  virtual ~Allocator() {}

  class AllocationDeleter {
   public:
    inline void operator()(Allocation* allocation) const {
      Allocator* allocator = allocation->TopDecoratedAllocator();
      allocator->Free(allocation);
    }
  };

  using AllocationPtr = std::unique_ptr<Allocation, AllocationDeleter>;

  // Allocate an allocation.
  // size may be 0, but it would be too complex if we handle size == 0
  // in each Allocator. So we handle size == 0 inside AllocatorFacade
  // in our design.
  inline AllocationPtr Allocate(size_t size) {
    auto ptr = AllocateImpl(size);
    ptr->RegisterDecoratedAllocator(this);
    return AllocationPtr(ptr);
  }

  // This function should not be called outside Allocator class
  inline void Free(Allocation* allocation) {
    allocation->PopDecoratedAllocator();
    FreeImpl(allocation);
  }

  inline uint64_t Release(const platform::Place& place) {
    return ReleaseImpl(place);
  }

  // True if the `Allocate` is thread safe.
  virtual bool IsAllocThreadSafe() const;

 protected:
  virtual Allocation* AllocateImpl(size_t size) = 0;
  virtual void FreeImpl(Allocation* allocation);
  virtual uint64_t ReleaseImpl(const platform::Place& place) { return 0; }
};

using AllocationDeleter = Allocator::AllocationDeleter;
using AllocationPtr = Allocator::AllocationPtr;

inline size_t AlignedSize(size_t size, size_t alignment) {
  auto remaining = size % alignment;
  return remaining == 0 ? size : size + alignment - remaining;
}

inline size_t AlignedPtrOffset(const void* ptr, size_t alignment) {
  auto ptr_addr = reinterpret_cast<uintptr_t>(ptr);
  auto diff = ptr_addr % alignment;
  return diff == 0 ? 0 : alignment - diff;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
