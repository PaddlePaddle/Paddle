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

#include "paddle/fluid/platform/temporary_allocator.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"

DEFINE_double(limit_of_temporary_allocation, -1,
              "The up limit of temporary_allocation size.");

namespace paddle {
namespace platform {
namespace alloc = memory::allocation;

TemporaryAllocation::TemporaryAllocation(
    alloc::AllocationPtr &&underlying_allocation)
    : Allocation(underlying_allocation->ptr(), underlying_allocation->size(),
                 underlying_allocation->place()),
      underlying_allocation_(std::move(underlying_allocation)) {}

TemporaryAllocator::TemporaryAllocator(platform::Place place) : place_(place) {
  temp_mem_queue_.reset(new std::deque<TemporaryAllocation *>());
}

bool TemporaryAllocator::IsAllocThreadSafe() const { return true; }

void TemporaryAllocator::Release(const std::function<void()> &callback) {
  std::shared_ptr<std::deque<TemporaryAllocation *>> t_allocations;
  {
    std::unique_lock<std::mutex> lock(mtx_);
    callback();
    t_allocations = temp_mem_queue_;
    temp_mem_queue_.reset(new std::deque<TemporaryAllocation *>());
    wait_delete_mem_ = 0;
  }
  for (auto tmp : *t_allocations) {
    VLOG(10) << "Delete temporary allocation " << tmp->ptr()
             << " size: " << tmp->size();
    delete tmp;
  }
}

void TemporaryAllocator::Free(alloc::Allocation *allocation) {
  auto *temp_allocation = dynamic_cast<TemporaryAllocation *>(allocation);
  PADDLE_ENFORCE_NOT_NULL(temp_allocation);
  if (platform::is_gpu_place(temp_allocation->place())) {
    size_t wait_delete_mem = 0;
    {
      std::unique_lock<std::mutex> lock(mtx_);
      temp_mem_queue_->emplace_back(temp_allocation);
      wait_delete_mem_ += temp_allocation->size();
      wait_delete_mem = wait_delete_mem_;
      VLOG(10) << "Move temporary allocation: " << temp_allocation->ptr()
               << " to delete queue: " << temp_allocation->size() << "; "
               << "wait_delete_mem: " << wait_delete_mem_;
    }
    if (FLAGS_limit_of_temporary_allocation > 0 &&
        wait_delete_mem > FLAGS_limit_of_temporary_allocation) {
      Release(callback_);
    }
    return;
  }
  delete temp_allocation;
}

size_t TemporaryAllocator::TemporaryAllocationQueueSize() {
  std::unique_lock<std::mutex> lock(mtx_);
  return temp_mem_queue_ ? temp_mem_queue_->size() : 0;
}

void TemporaryAllocator::SetCallback(const std::function<void()> &callback) {
  callback_ = callback;
}

alloc::Allocation *TemporaryAllocator::AllocateImpl(
    size_t size, alloc::Allocator::Attr attr) {
  auto raw_allocation =
      alloc::AllocatorFacade::Instance().Alloc(place_, size, attr);
  auto temp_mem = new TemporaryAllocation(std::move(raw_allocation));
  VLOG(10) << "Alloc temporary allocation: " << temp_mem->ptr() << ": " << size;
  return temp_mem;
}

}  // namespace platform
}  // namespace paddle
