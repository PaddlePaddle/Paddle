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

#include "paddle/fluid/memory/allocation/buffered_allocator.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace paddle {
namespace memory {
namespace allocation {

BufferedAllocator::BufferedAllocator(std::unique_ptr<Allocator>&& allocator) {
  std::vector<size_t> division_plan(8 * sizeof(size_t));
  for (size_t i = 0; i < 8 * sizeof(size_t); ++i) {
    division_plan[i] = (static_cast<size_t>(1) << i);
  }
  InitAndEnforceCheck(std::move(allocator), division_plan);
}

BufferedAllocator::BufferedAllocator(std::unique_ptr<Allocator>&& allocator,
                                     const std::vector<size_t>& division_plan) {
  InitAndEnforceCheck(std::move(allocator), division_plan);
}

BufferedAllocator::~BufferedAllocator() {
  for (auto& v : allocations_) {
    for (auto& pair : v) {
      underlying_allocator_->FreeUniquePtr(std::move(pair.second));
    }
  }
}

void BufferedAllocator::InitAndEnforceCheck(
    std::unique_ptr<Allocator>&& allocator,
    const std::vector<size_t>& division_plan) {
  underlying_allocator_.reset(
      dynamic_cast<UnmanagedAllocator*>(allocator.release()));
  PADDLE_ENFORCE_NOT_NULL(
      underlying_allocator_,
      "Underlying allocator of BufferedAllocator must be unmanaged");
  if (underlying_allocator_->IsAllocThreadSafe()) {
    mtx_.reset(new std::mutex());
  }
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  if (division_plan.empty()) {
    division_plan_.assign({0, kMax});
  } else {
    auto from = division_plan.front() == 0 ? division_plan.begin() + 1
                                           : division_plan.begin();
    auto to = division_plan.back() == kMax ? division_plan.end() - 1
                                           : division_plan.end();
    division_plan_.reserve(to - from + 2);
    division_plan_.push_back(0);
    division_plan_.insert(division_plan_.end(), from, to);
    division_plan_.push_back(kMax);
    for (size_t i = 1; i < division_plan_.size(); ++i) {
      PADDLE_ENFORCE_LT(division_plan_[i - 1], division_plan_[i],
                        "Division plan must be strictly sorted");
    }
  }
  allocations_.resize(division_plan_.size() - 1);
}

void BufferedAllocator::InsertAllocationImpl(
    std::unique_ptr<Allocation>&& allocation) {
  auto size = allocation->size();
  auto idx = GetListIndex(size);
  allocations_[idx].insert(std::pair<size_t, std::unique_ptr<Allocation>>(
      size, std::move(allocation)));
}

void BufferedAllocator::InsertAllocation(
    std::unique_ptr<Allocation>&& allocation) {
  if (mtx_) {
    std::lock_guard<std::mutex> lock(*mtx_);
    InsertAllocationImpl(std::move(allocation));
  } else {
    InsertAllocationImpl(std::move(allocation));
  }
}

bool BufferedAllocator::Match(const std::unique_ptr<Allocation>& allocation,
                              size_t size) {
  return (allocation->size() >> 1) <= size;
}

size_t BufferedAllocator::GetListIndex(size_t size) {
  auto it =
      std::upper_bound(division_plan_.begin(), division_plan_.end(), size);
  return static_cast<size_t>(it - division_plan_.begin()) - 1;
}

std::unique_ptr<Allocation> BufferedAllocator::RemoveAllocationImpl(
    size_t size) {
  auto idx = GetListIndex(size);
  auto& allocation_map = allocations_[idx];
  auto it = allocation_map.lower_bound(size);
  // Only remove allocation whose size is not more than twice of requested size
  if (it != allocation_map.end() && Match(it->second, size)) {
    auto ret = std::move(it->second);
    allocation_map.erase(it);
    return ret;
  } else {
    return nullptr;
  }
}

std::unique_ptr<Allocation> BufferedAllocator::RemoveAllocation(size_t size) {
  if (mtx_) {
    std::lock_guard<std::mutex> lock(*mtx_);
    return RemoveAllocationImpl(size);
  } else {
    return RemoveAllocationImpl(size);
  }
}

std::unique_ptr<Allocation> BufferedAllocator::Allocate(size_t size,
                                                        Allocator::Attr attr) {
  auto ret = RemoveAllocation(size);
  if (!ret) {
    try {
      return underlying_allocator_->Allocate(size, attr);
    } catch (BadAlloc&) {
      // if allocation failed, try to free some memorys from buffers
      FreeAllocations(size);
      return underlying_allocator_->Allocate(size, attr);
    }
  }
  return ret;
}

void BufferedAllocator::FreeAllocationsImpl(size_t size) {
  if (UNLIKELY(size == 0)) return;
  size_t cur = 0;
  for (auto& alloc_map : allocations_) {
    // use reverse iterator to free large allocations first
    while (!alloc_map.empty()) {
      auto it = --(alloc_map.end());
      cur += it->second->size();
      underlying_allocator_->FreeUniquePtr(std::move(it->second));
      alloc_map.erase(it);
      if (cur >= size) return;
    }
  }
}

void BufferedAllocator::FreeAllocations(size_t size) {
  if (mtx_) {
    std::lock_guard<std::mutex> lock(*mtx_);
    FreeAllocationsImpl(size);
  } else {
    FreeAllocationsImpl(size);
  }
}

void BufferedAllocator::FreeUniquePtr(std::unique_ptr<Allocation> allocation) {
  InsertAllocation(std::move(allocation));
}

bool BufferedAllocator::IsAllocThreadSafe() const { return mtx_ != nullptr; }

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
