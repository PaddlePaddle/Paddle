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

class VMMAutoGrowthBestFitMultiPoolAllocationV2
    : public Allocation,
      public VMMRemapEventAllocation {
 public:
  VMMAutoGrowthBestFitMultiPoolAllocationV2(
      AllocationPtr underlying_allocation,
      VMMAutoGrowthBestFitAllocatorV2* allocator,
      PoolType pool_type)
      : Allocation(
            underlying_allocation->ptr(),
            static_cast<Allocation*>(underlying_allocation.get())->base_ptr(),
            underlying_allocation->size(),
            underlying_allocation->place()),
        underlying_allocation_(std::move(underlying_allocation)),
        allocator_(allocator),
        pool_type_(pool_type),
        remap_allocation_(dynamic_cast<VMMRemapEventAllocation*>(
            underlying_allocation_.get())) {}

  AllocationPtr TakeUnderlyingAllocation() {
    return std::move(underlying_allocation_);
  }

  VMMAutoGrowthBestFitAllocatorV2* allocator() const { return allocator_; }
  PoolType pool_type() const { return pool_type_; }
  bool SetVMMRemapEvent(gpuStream_t stream,
                        std::shared_ptr<CUDAEventGuard> event) override {
    if (remap_allocation_ == nullptr) {
      return false;
    }
    return remap_allocation_->SetVMMRemapEvent(stream, std::move(event));
  }

 private:
  AllocationPtr underlying_allocation_;
  VMMAutoGrowthBestFitAllocatorV2* allocator_;
  PoolType pool_type_;
  VMMRemapEventAllocation* remap_allocation_{nullptr};
};

}  // namespace

VMMAutoGrowthBestFitMultiPoolAllocatorV2::
    VMMAutoGrowthBestFitMultiPoolAllocatorV2(
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>& small_allocator,
        const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>& large_allocator,
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
  return new VMMAutoGrowthBestFitMultiPoolAllocationV2(  // NOLINT
      std::move(allocation),
      route.allocator,
      route.pool_type);
}

size_t VMMAutoGrowthBestFitMultiPoolAllocatorV2::CompactImpl(
    const Place& place, size_t requested_size) {
  if (requested_size > 0) {
    const auto route = RouteAllocation(requested_size);
    if (route.pool_type == PoolType::kSmall) {
      return 0;
    }
    return route.allocator->Compact(place, requested_size);
  }
  // Compact/remap targets the large pool only. Small-pool requests are cheap
  // to satisfy through normal reuse/grow and do not justify remap overhead.
  return large_allocator_->Compact(place, requested_size);
}

void VMMAutoGrowthBestFitMultiPoolAllocatorV2::FreeImpl(
    phi::Allocation* allocation) {
  auto* wrapped_allocation =
      static_cast<VMMAutoGrowthBestFitMultiPoolAllocationV2*>(allocation);
  auto* allocator = wrapped_allocation->allocator();
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      common::errors::NotFound(
          "No VMM pool allocator found for pool %d.",
          static_cast<int>(wrapped_allocation->pool_type())));
  auto underlying_allocation = wrapped_allocation->TakeUnderlyingAllocation();
  allocator->Free(underlying_allocation.release());
  delete wrapped_allocation;
}

void VMMAutoGrowthBestFitMultiPoolAllocatorV2::GetFreeBlockStats(
    size_t* total_free, size_t* max_free, size_t alloc_size) {
  if (alloc_size > 0) {
    // Route-aware query: return stats only for the target pool.
    const auto route = RouteAllocation(alloc_size);
    route.allocator->GetFreeBlockStats(total_free, max_free);
    return;
  }
  // Legacy aggregate path (alloc_size == 0).
  size_t s_total = 0, s_max = 0, l_total = 0, l_max = 0;
  small_allocator_->GetFreeBlockStats(&s_total, &s_max);
  large_allocator_->GetFreeBlockStats(&l_total, &l_max);
  *total_free = s_total + l_total;
  *max_free = std::max(s_max, l_max);
}

bool VMMAutoGrowthBestFitMultiPoolAllocatorV2::SetBlockRemapEvent(
    void* ptr, gpuStream_t stream, std::shared_ptr<CUDAEventGuard> event) {
  if (small_allocator_->SetBlockRemapEvent(ptr, stream, event)) {
    return true;
  }
  return large_allocator_->SetBlockRemapEvent(ptr, stream, event);
}

uint64_t VMMAutoGrowthBestFitMultiPoolAllocatorV2::ReleaseImpl(
    const Place& place) {
  return small_allocator_->Release(place) + large_allocator_->Release(place);
}

VMMAutoGrowthBestFitMultiPoolAllocatorV2::AllocationRoute
VMMAutoGrowthBestFitMultiPoolAllocatorV2::RouteAllocation(size_t size) const {
  const size_t routed_size = AlignedSize(size, small_allocator_->alignment());
  if (routed_size < small_allocation_threshold_) {
    return {PoolType::kSmall, small_allocator_.get()};
  }
  return {PoolType::kLarge, large_allocator_.get()};
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
