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

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/inlined_vector.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/enforce.h"

#ifdef PADDLE_WITH_NCCL
#include <nccl.h>
#include "paddle/phi/backends/dynload/nccl.h"
#endif

COMMON_DECLARE_string(allocator_strategy);
COMMON_DECLARE_bool(sync_after_alloc);
COMMON_DECLARE_int64(alloc_fill_value);

namespace paddle {
namespace memory {
namespace allocation {

// Exception when `Alloc`/`AllocShared` failed
struct BadAlloc : public std::exception {
  inline explicit BadAlloc(std::string err_msg, const char* file, int line)
      : err_str_(platform::GetCompleteTraceBackString(
            std::move(err_msg), file, line)) {}

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
 * Therefore, `decorated_allocators_` of the new Allocation object
 * would
 * be a new chain, differing from the underlying Allocation object.
 */
class Allocation : public phi::Allocation {
 public:
  Allocation(void* ptr, size_t size, phi::Place place)
      : phi::Allocation(ptr, size, place), base_ptr_(ptr) {}
  Allocation(void* ptr, void* base_ptr, size_t size, const phi::Place& place)
      : phi::Allocation(ptr, size, place), base_ptr_(base_ptr) {}

  void* base_ptr() const { return base_ptr_; }

 private:
  inline void RegisterDecoratedAllocator(Allocator* allocator) {
    decorated_allocators_.emplace_back(allocator);
  }

  inline void PopDecoratedAllocator() { decorated_allocators_.pop_back(); }

  inline Allocator* TopDecoratedAllocator() {
    return decorated_allocators_.back();
  }

 private:
  void* base_ptr_;  // the point that directly requested from system

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

using AllocationPtr = phi::Allocator::AllocationPtr;
using DecoratedAllocationPtr =
    std::unique_ptr<Allocation, phi::Allocator::DeleterType>;

template <typename T>
static T&& FillValue(T&& allocation) {
#if defined(PADDLE_WITH_CUDA)
  if (allocation != nullptr) {
    if (FLAGS_sync_after_alloc || FLAGS_alloc_fill_value >= 0) {
      bool need_sync = !phi::is_cpu_place(allocation->place());
      if (need_sync) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
      }
      if (FLAGS_alloc_fill_value >= 0) {
        VLOG(10) << "Set " << FLAGS_alloc_fill_value << " on "
                 << allocation->ptr() << " " << allocation->place() << " "
                 << allocation->size();
        if (phi::is_gpu_place(allocation->place())) {
          PADDLE_ENFORCE_GPU_SUCCESS(cudaMemset(
              allocation->ptr(), FLAGS_alloc_fill_value, allocation->size()));
        } else {
          std::memset(
              allocation->ptr(), FLAGS_alloc_fill_value, allocation->size());
        }
        if (need_sync) {
          PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
        }
      }
    }
  }
#endif
  return std::forward<T>(allocation);
}

// Base interface class of memory Allocator.
class Allocator : public phi::Allocator {
 public:
  static void AllocationDeleter(phi::Allocation* allocation) {
    Allocator* allocator =
        static_cast<Allocation*>(allocation)->TopDecoratedAllocator();
    allocator->Free(allocation);
  }

  // Allocate an allocation.
  // size may be 0, but it would be too complex if we handle size == 0
  // in each Allocator. So we handle size == 0 inside AllocatorFacade
  // in our design.
  AllocationPtr Allocate(size_t size) override {
    auto ptr = AllocateImpl(size);
    static_cast<Allocation*>(ptr)->RegisterDecoratedAllocator(this);
    return FillValue(AllocationPtr(ptr, AllocationDeleter));
  }

  void Free(phi::Allocation* allocation) {
    static_cast<Allocation*>(allocation)->PopDecoratedAllocator();
    FreeImpl(allocation);
  }

  uint64_t Release(const phi::Place& place) { return ReleaseImpl(place); }

 protected:
  virtual phi::Allocation* AllocateImpl(size_t size) = 0;
  virtual void FreeImpl(phi::Allocation* allocation);
  virtual uint64_t ReleaseImpl(const phi::Place& place UNUSED) { return 0; }
};

inline size_t AlignedSize(size_t size, size_t alignment) {
  auto remaining = size % alignment;  // NOLINT
  return remaining == 0 ? size : size + alignment - remaining;
}

inline size_t AlignedPtrOffset(const void* ptr, size_t alignment) {
  auto ptr_addr = reinterpret_cast<uintptr_t>(ptr);
  auto diff = ptr_addr % alignment;
  return diff == 0 ? 0 : alignment - diff;
}

template <typename Derived, typename Base, typename BaseDel>
decltype(auto) static_unique_ptr_cast(std::unique_ptr<Base, BaseDel>&& p) {
  static_assert(std::is_base_of<Base, Derived>::value,
                "Derived type must derive from Base.");
  auto d = static_cast<Derived*>(p.release());
  return std::unique_ptr<Derived, BaseDel>(d, p.get_deleter());
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
