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

#include "paddle/fluid/framework/inlined_vector.h"
#include "paddle/pten/core/allocator.h"

DECLARE_string(allocator_strategy);

namespace paddle {
namespace memory {
namespace allocation {
namespace experimental {
class Allocator;
}

class AllocationContext {
 public:
  explicit AllocationContext(void* base_ptr) : base_ptr_(base_ptr) {}
  virtual ~AllocationContext() = default;

  static void Deleter(pten::Allocation* allocation) {
    auto* ctx = allocation->CastContext<AllocationContext>(Deleter);
    auto* allocator = ctx->TopDecoratedAllocator();
    allocator->Free(allocation);
    delete ctx;
  }

  inline void* base_ptr() const {
    PADDLE_ENFORCE_EQ(FLAGS_allocator_strategy, "auto_growth",
                      paddle::platform::errors::Unimplemented(
                          "base_ptr() is only implemented for auto_growth "
                          "strategy, not support %s strategy",
                          FLAGS_allocator_strategy));
    return base_ptr_;
  }

 protected:
  void RegisterDecoratedAllocator(experimental::Allocator* allocator) {
    decorated_allocators_.emplace_back(allocator);
  }

  void PopDecoratedAllocator() { decorated_allocators_.pop_back(); }

  experimental::Allocator* TopDecoratedAllocator() {
    return decorated_allocators_.back();
  }

 private:
  friend class experimental::Allocator;
  void* base_ptr_{nullptr};
  static constexpr size_t kReserveAllocatorNum = 8;
  using DecoratedAllocatorStack =
      framework::InlinedVector<experimental::Allocator*, kReserveAllocatorNum>;

  DecoratedAllocatorStack decorated_allocators_;
};

namespace experimental {

class Allocator : public pten::Allocator {
 public:
  pten::Allocation Allocate(size_t size) {
    auto allocation = AllocateImpl(size);
    auto* ctx =
        allocation.CastContext<AllocationContext>(AllocationContext::Deleter);
    ctx->RegisterDecoratedAllocator(this);
    return allocation;
  }

  void Free(pten::Allocation* allocation) {
    auto* ctx =
        allocation->CastContext<AllocationContext>(AllocationContext::Deleter);
    ctx->PopDecoratedAllocator();
    FreeImpl(allocation);
  }

  uint64_t Release(const platform::Place& place) { return ReleaseImpl(place); }

  // True if the `Allocate` is thread safe.
  virtual bool IsAllocThreadSafe() const { return false; }

 protected:
  virtual pten::Allocation AllocateImpl(size_t size) = 0;
  virtual void FreeImpl(pten::Allocation* allocation) {
    auto* ctx =
        allocation->CastContext<AllocationContext>(AllocationContext::Deleter);
    Allocator* allocator = ctx->TopDecoratedAllocator();
    allocator->Free(allocation);
  }
  virtual uint64_t ReleaseImpl(const platform::Place& place) { return 0; }
};
}  // namespace experimental
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
