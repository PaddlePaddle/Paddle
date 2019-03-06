// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/multi_bin_buffered_allocator.h"
#include <algorithm>
#include <limits>
#include "paddle/fluid/platform/lock_guard_ptr.h"

DEFINE_double(tolerant_times, 2,
              "Tolerant memory size times of buffered_allocator");

namespace paddle {
namespace memory {
namespace allocation {

static void CheckAndModifyMemoryDivisionPlan(
    std::vector<size_t> *division_plan) {
  // Check whether the division plan is strictly sorted
  bool is_strictly_sorted = true;
  for (size_t i = 1; i < division_plan->size(); ++i) {
    if ((*division_plan)[i - 1] >= (*division_plan)[i]) {
      is_strictly_sorted = false;
      break;
    }
  }
  PADDLE_ENFORCE(is_strictly_sorted, "Divison plan must be stricted sorted");

  // Insert 0 and remove MAX to disivion plan for clean binary searching code
  if (division_plan->empty() || division_plan->front() != 0) {
    division_plan->insert(division_plan->begin(), 0);
  }

  constexpr auto kSizeTypeMax = std::numeric_limits<size_t>::max();
  if (division_plan->back() == kSizeTypeMax) {
    division_plan->pop_back();
  }

  PADDLE_ENFORCE(division_plan->size() >= 1, "Division plan cannot be empty");
}

static std::vector<size_t> GetDefaultDivisionPlan() {
  std::vector<size_t> plan;
  for (size_t i = 0; i < sizeof(size_t) * 8; ++i) {
    plan.push_back(static_cast<size_t>(1) << i);
  }
  return plan;
}

inline static size_t FindDivisionPlanBinIndex(const std::vector<size_t> &bins,
                                              size_t size) {
  return static_cast<size_t>(std::upper_bound(bins.begin(), bins.end(), size) -
                             bins.begin() - 1);
}

inline static size_t TolerantUpperSize(size_t size) {
  return static_cast<size_t>(size * FLAGS_tolerant_times);
}

MultiBinBufferedAllocator::MultiBinBufferedAllocator(
    std::shared_ptr<Allocator> underlying_allocator)
    : MultiBinBufferedAllocator(std::move(underlying_allocator),
                                GetDefaultDivisionPlan()) {}

MultiBinBufferedAllocator::MultiBinBufferedAllocator(
    std::shared_ptr<Allocator> underlying_allocator,
    const std::vector<size_t> &division_plan)
    : underlying_allocator_(std::move(underlying_allocator)),
      division_plan_(division_plan) {
  CheckAndModifyMemoryDivisionPlan(&division_plan_);
  allocations_.resize(division_plan_.size());
  mtx_.resize(division_plan_.size());
  if (underlying_allocator_->IsAllocThreadSafe()) {
    for (auto &mtx : mtx_) {
      mtx.reset(new std::mutex());
    }
  }

  VLOG(1) << "FLAGS_tolerant_times = " << FLAGS_tolerant_times;
}

void MultiBinBufferedAllocator::FreeImpl(Allocation *allocation) {
  auto bin_index = FindDivisionPlanBinIndex(division_plan_, allocation->size());
  {
    platform::LockGuardPtr<std::mutex> guard(mtx_[bin_index]);
    allocations_[bin_index].emplace(allocation->size(),
                                    AllocationPtr(allocation));
  }
}

void MultiBinBufferedAllocator::FreeCache(size_t size, size_t bin_index) {
  size_t accumulated_size = 0;
  // FIXME(zjl): free the largest first when there is no extra
  for (size_t i = allocations_.size() - 1; i != static_cast<size_t>(-1); --i) {
    platform::LockGuardPtr<std::mutex> lock(mtx_[i]);
    if (allocations_[i].empty()) continue;
    auto it = --allocations_[i].end();
    do {
      accumulated_size += it->second->size();
      underlying_allocator_->Free(it->second.release());
      allocations_[i].erase(it--);
      if (accumulated_size >= size) {
        return;
      }
    } while (!allocations_[i].empty());
  }
}

Allocation *MultiBinBufferedAllocator::AllocateImpl(size_t size, Attr attr) {
  auto bin_index = FindDivisionPlanBinIndex(division_plan_, size);
  auto upper_size = TolerantUpperSize(size);

  for (; upper_size >= division_plan_[bin_index]; ++bin_index) {
    auto &allocation = allocations_[bin_index];
    platform::LockGuardPtr<std::mutex> lock(mtx_[bin_index]);
    auto it = allocation.lower_bound(size);
    if (it != allocation.end() && it->second->size() < upper_size) {
      auto ret = std::move(it->second);
      allocation.erase(it);
      return ret.release();
    }
  }

  try {
    return underlying_allocator_->Allocate(size, attr).release();
  } catch (BadAlloc &) {
    VLOG(2) << "BadAlloc raises, try to free " << size << " caches";
    FreeCache(size, bin_index);
    return underlying_allocator_->Allocate(size, attr).release();
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
