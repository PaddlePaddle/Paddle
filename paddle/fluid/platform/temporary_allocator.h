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
#include <condition_variable>  // NOLINT
#include <deque>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/lock_guard_ptr.h"
namespace paddle {
namespace platform {

class TemporaryAllocation : public memory::allocation::Allocation {
 public:
  explicit TemporaryAllocation(
      memory::allocation::AllocationPtr &&underlying_allocation);

  memory::allocation::AllocationPtr underlying_allocation_;
};

/*! \brief the TemporaryAllocator is used to alloc the temporary allocation
 * which used by CUDA's async operation.
 *
 * The TemporaryAllocator contains a temp_allocation_queue which
 * is used to store the temporary allocations. The allocation, which is
 * allocated by TemporaryAllocator, is a unique_ptr, and when it is not held
 * by any variable, it will be pushed into the  temp_allocation_queue.
 *
 * There is one opportunity to free the allocations of temp_allocation_queue:
 *   - when the allocation size of opportunities exceeds a certain threshold
 *     (defined by FLAGS_limit_of_tmp_allocation).
 *
 * */
class TemporaryAllocator : public memory::allocation::Allocator {
 public:
  explicit TemporaryAllocator(platform::Place place);

  void Release(const std::function<void()> &callback);

  size_t TemporaryAllocationQueueSize();

  bool IsAllocThreadSafe() const override;

  void SetCallback(const std::function<void()> &callback);

 protected:
  void Free(memory::allocation::Allocation *allocation) override;

  memory::allocation::Allocation *AllocateImpl(
      size_t size, memory::allocation::Allocator::Attr attr) override;

 private:
  platform::Place place_;
  // When the allocation is not held by any variable, it should be placed
  // to temp_mem_map immediately.
  std::unique_ptr<std::multimap<size_t, TemporaryAllocation *>> temp_mem_map_{
      nullptr};
  std::mutex mtx_;
  size_t wait_delete_mem_{0};
  std::function<void()> callback_;
};

}  // namespace platform
}  // namespace paddle
