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

#include <atomic>  // NOLINT
#include <functional>
#include <memory>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT
#include <vector>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

// The AutoIncrementAllocator manages many underlying allocators. If none of
// them can allocate the request memory, a new allocator will be created and
// invoke its `allocate` method.
//
// NOTE(yy): The AutoIncrementAllocator will prefer to allocate memory from
// the latest successful allocator.
//
// NOTE(yy): We may need to release an underlying allocator if it allocate
// nothing. However, it is generally not useful, since it will make performance
// undetermined.
//
// NOTE(yy): This allocator is only locked when creating new underlying
// allocator. The allocation requests from many threads may be dispatched
// to the same underlying allocator. So the underlying allocator must be
// thread safe.
//
// NOTE(zjl): Add capacity parameters to constructor. A high-performance
// thread-safe std::vector with varying size is hard to implement.
// Fortunately, we can get the total GPU memory and each chunk size.
// Therefore, we can get the suitable capacity of AutoIncrementAllocator.
class AutoIncrementAllocator : public Allocator {
 public:
  // Creator is the method to create ManagedAllocator
  using AllocatorCreator = std::function<std::shared_ptr<Allocator>()>;

  explicit AutoIncrementAllocator(AllocatorCreator&& creator, size_t capacity)
      : creator_(std::move(creator)), underlying_allocators_(capacity) {}

  bool IsAllocThreadSafe() const override;

 private:
  std::shared_ptr<Allocator> CreateNewAllocator();

 protected:
  Allocation* AllocateImpl(size_t size, Allocator::Attr attr) override;

 private:
  AllocatorCreator creator_;

  std::vector<AllocatorCreator::result_type> underlying_allocators_;
  std::atomic<size_t> allocator_num_{0};

  // Use std::atomic rather than std::mutex, since std::atomic is usually
  // lock-free
  std::atomic<size_t> prev_success_allocator_{0};

  std::mutex mtx_;
};
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
