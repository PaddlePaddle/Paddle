// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_multi_pool_allocator_v2.h"

#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

template <typename Map, typename Key, typename Value>
void EmplaceOrEnforce(Map* map,
                      Key&& key,
                      Value&& value,
                      const char* map_name) {
  const bool inserted =
      map->try_emplace(std::forward<Key>(key), std::forward<Value>(value))
          .second;
  PADDLE_ENFORCE_EQ(
      inserted,
      true,
      common::errors::AlreadyExists(
          "Duplicate key inserted into %s, allocator state is inconsistent.",
          map_name));
}

}  // namespace

VMMAutoGrowthBestFitMultiPoolAllocatorV2::
    VMMAutoGrowthBestFitMultiPoolAllocatorV2(
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
            small_allocator,
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>&
            large_allocator,
        size_t small_allocation_threshold,
        const GPUPlace& place)
    : small_allocator_(small_allocator),
      large_allocator_(large_allocator),
      small_allocation_threshold_(small_allocation_threshold),
      place_(place) {}

phi::Allocation* VMMAutoGrowthBestFitMultiPoolAllocatorV2::AllocateImpl(
    size_t size) {
  const auto route = RouteAllocation(size);
  PADDLE_ENFORCE_NOT_NULL(
      route.allocator,
      common::errors::NotFound("No VMM pool allocator found for pool %d.",
                               static_cast<int>(route.pool_type)));
  auto allocation = route.allocator->Allocate(size);
  {
    std::lock_guard<SpinLock> guard(spinlock_);
    EmplaceOrEnforce(&active_allocations_,
                     allocation->ptr(),
                     AllocationRoute{route.pool_type, route.allocator},
                     "active_allocations_");
  }
  return allocation.release();
}

void VMMAutoGrowthBestFitMultiPoolAllocatorV2::FreeImpl(
    phi::Allocation* allocation) {
  AllocationRoute route{PoolType::kLarge, nullptr};
  {
    std::lock_guard<SpinLock> guard(spinlock_);
    auto it = active_allocations_.find(allocation->ptr());
    PADDLE_ENFORCE_NE(
        it,
        active_allocations_.end(),
        common::errors::NotFound(
            "No VMM pool routing metadata found for allocation %p.",
            allocation->ptr()));
    route = it->second;
    active_allocations_.erase(it);
  }
  PADDLE_ENFORCE_NOT_NULL(
      route.allocator,
      common::errors::NotFound("No VMM pool allocator found for pool %d.",
                               static_cast<int>(route.pool_type)));
  route.allocator->Free(allocation);
}

bool VMMAutoGrowthBestFitMultiPoolAllocatorV2::SetBlockRemapEvent(
    void* ptr,
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    gpuStream_t stream,
    gpuEvent_t event
#else
    void* stream,
    void* event
#endif
) {
  AllocationRoute route{PoolType::kLarge, nullptr};
  {
    std::lock_guard<SpinLock> guard(spinlock_);
    auto it = active_allocations_.find(ptr);
    if (it == active_allocations_.end()) {
      return false;
    }
    route = it->second;
  }
  PADDLE_ENFORCE_NOT_NULL(
      route.allocator,
      common::errors::NotFound("No VMM pool allocator found for pool %d.",
                               static_cast<int>(route.pool_type)));
  return route.allocator->SetBlockRemapEvent(ptr, stream, event);
}

void VMMAutoGrowthBestFitMultiPoolAllocatorV2::ExportForIpc() {
  PADDLE_THROW(common::errors::Unimplemented(
      "VMM V2 does not support IPC yet, set "
      "FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=0 or wait for W5."));
}

void VMMAutoGrowthBestFitMultiPoolAllocatorV2::ImportFromIpc() {
  PADDLE_THROW(common::errors::Unimplemented(
      "VMM V2 does not support IPC yet, set "
      "FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=0 or wait for W5."));
}

uint64_t VMMAutoGrowthBestFitMultiPoolAllocatorV2::ReleaseImpl(
    const Place& place) {
  return small_allocator_->Release(place) + large_allocator_->Release(place);
}

VMMAutoGrowthBestFitMultiPoolAllocatorV2::AllocationRoute
VMMAutoGrowthBestFitMultiPoolAllocatorV2::RouteAllocation(size_t size) const {
  if (size < small_allocation_threshold_) {
    return {PoolType::kSmall, small_allocator_.get()};
  }
  return {PoolType::kLarge, large_allocator_.get()};
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
