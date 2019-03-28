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
#include <memory>
#include "paddle/fluid/memory/allocation/allocator_facade.h"

DEFINE_int64(limit_of_tmp_allocation, -1,
             "The up limit of temporary_allocation size.");
DEFINE_double(times_excess_than_required_tmp_allocation, 2,
              "times_excess_than_required_tmp_allocation indicates the "
              "max size the TemporaryAllocator can return. For example, "
              "if the required memory size is N, and "
              "times_excess_than_required_tmp_allocation is 2.0, "
              "the TemporaryAllocator will return the available allocation "
              "that the range of size is N ~ 2*N.");

namespace paddle {
namespace platform {
namespace alloc = memory::allocation;

TemporaryAllocation::TemporaryAllocation(
    alloc::AllocationPtr &&underlying_allocation)
    : Allocation(underlying_allocation->ptr(), underlying_allocation->size(),
                 underlying_allocation->place()),
      underlying_allocation_(std::move(underlying_allocation)) {}

TemporaryAllocator::TemporaryAllocator(platform::Place place) : place_(place) {
  temp_mem_map_.reset(new std::multimap<size_t, TemporaryAllocation *>());
}

bool TemporaryAllocator::IsAllocThreadSafe() const { return true; }

void TemporaryAllocator::Release(const std::function<void()> &callback) {
  std::unique_ptr<std::multimap<size_t, TemporaryAllocation *>> t_allocations;
  {
    std::unique_lock<std::mutex> lock(mtx_);
    callback();
    t_allocations.swap(temp_mem_map_);
    temp_mem_map_.reset(new std::multimap<size_t, TemporaryAllocation *>());
    wait_delete_mem_ = 0;
  }

  for (auto tmp : *t_allocations) {
    VLOG(10) << "Delete temporary allocation " << tmp.second->ptr()
             << " size: " << tmp.second->size();
    delete tmp.second;
  }
}

void TemporaryAllocator::Free(alloc::Allocation *allocation) {
  auto *temp_allocation = dynamic_cast<TemporaryAllocation *>(allocation);
  PADDLE_ENFORCE_NOT_NULL(temp_allocation);
  if (platform::is_gpu_place(temp_allocation->place())) {
    PADDLE_ENFORCE(platform::is_same_place(temp_allocation->place(), place_),
                   "The place should be the same.");
    size_t wait_delete_mem = 0;
    {
      std::unique_lock<std::mutex> lock(mtx_);
      temp_mem_map_->emplace(temp_allocation->size(), temp_allocation);
      wait_delete_mem_ += temp_allocation->size();
      wait_delete_mem = wait_delete_mem_;
      VLOG(10) << "Move temporary allocation: " << temp_allocation->ptr()
               << " to delete queue: " << temp_allocation->size() << "; "
               << "wait_delete_mem: " << wait_delete_mem;
    }

    if (FLAGS_limit_of_tmp_allocation > 0 &&
        wait_delete_mem > static_cast<size_t>(FLAGS_limit_of_tmp_allocation)) {
      PADDLE_ENFORCE(callback_ != nullptr, "The callback is non-initialized.");
      Release(callback_);
    }
    return;
  }
  VLOG(10) << "Delete temporary allocation " << temp_allocation->ptr()
           << " size: " << temp_allocation->size();
  delete temp_allocation;
}

size_t TemporaryAllocator::TemporaryAllocationQueueSize() {
  std::unique_lock<std::mutex> lock(mtx_);
  return temp_mem_map_ ? temp_mem_map_->size() : 0;
}

void TemporaryAllocator::SetCallback(const std::function<void()> &callback) {
  callback_ = callback;
}

alloc::Allocation *TemporaryAllocator::AllocateImpl(
    size_t size, alloc::Allocator::Attr attr) {
  {
    // Find available allocation in temp_mem_map.
    std::unique_lock<std::mutex> lock(mtx_);
    if (temp_mem_map_->size()) {
      auto it = temp_mem_map_->lower_bound(size);
      // FIXME(zcd): Not sure the best value of excess fraction.
      if (it != temp_mem_map_->end() &&
          it->first <
              static_cast<size_t>(
                  size * FLAGS_times_excess_than_required_tmp_allocation)) {
        auto tmp_ptr = it->second;
        temp_mem_map_->erase(it);
        wait_delete_mem_ -= tmp_ptr->size();
        VLOG(10) << "Reuse temporary allocation: " << tmp_ptr->ptr() << ": "
                 << tmp_ptr->size();
        return tmp_ptr;
      }
    }
  }
  // If not find the the available allocation, get allocation from
  // AllocatorFacadeInstance.
  auto raw_allocation =
      alloc::AllocatorFacade::Instance().Alloc(place_, size, attr);
  auto temp_mem = new TemporaryAllocation(std::move(raw_allocation));
  VLOG(10) << "Alloc temporary allocation: " << temp_mem->ptr() << ": " << size;
  return temp_mem;
}

}  // namespace platform
}  // namespace paddle
