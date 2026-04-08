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

#include "gtest/gtest.h"

#define private public
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_multi_pool_allocator_v2.h"
#undef private

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2> CreatePoolAllocator(
    size_t handle_size, PoolType pool_type) {
  auto underlying = std::make_shared<CUDAVirtualMemAllocatorV2>(
      phi::GPUPlace(), handle_size, pool_type);
  return std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
      underlying, 256, phi::GPUPlace(), pool_type);
}

std::unique_ptr<VMMAutoGrowthBestFitMultiPoolAllocatorV2> CreateAllocator() {
  return std::make_unique<VMMAutoGrowthBestFitMultiPoolAllocatorV2>(
      CreatePoolAllocator(2UL << 20, PoolType::kSmall),
      CreatePoolAllocator(2UL << 20, PoolType::kLarge),
      2UL << 20,
      phi::GPUPlace());
}

}  // namespace

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2, RouteSmallAndLarge) {
  auto allocator = CreateAllocator();

  auto small = allocator->Allocate(256UL);
  auto large = allocator->Allocate(2UL << 20);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(large, nullptr);

  EXPECT_EQ(allocator->active_allocations_[small->ptr()].pool_type,
            PoolType::kSmall);
  EXPECT_EQ(allocator->active_allocations_[large->ptr()].pool_type,
            PoolType::kLarge);
  EXPECT_EQ(allocator->active_allocations_[small->ptr()].allocator,
            allocator->small_allocator_.get());
  EXPECT_EQ(allocator->active_allocations_[large->ptr()].allocator,
            allocator->large_allocator_.get());

  EXPECT_EQ(allocator->small_allocator_->allocated_blocks_.size(), 1UL);
  EXPECT_EQ(allocator->large_allocator_->allocated_blocks_.size(), 1UL);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2, SetBlockRemapEventRoutesByPtr) {
  auto allocator = CreateAllocator();

  auto allocation = allocator->Allocate(256UL);
  ASSERT_NE(allocation, nullptr);

  gpuEvent_t event = nullptr;
  ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            cudaSuccess);
  auto* ptr = allocation->ptr();
  ASSERT_TRUE(allocator->SetBlockRemapEvent(ptr, nullptr, event));

  auto it = allocator->small_allocator_->allocated_blocks_.find(ptr);
  ASSERT_NE(it, allocator->small_allocator_->allocated_blocks_.end());
  EXPECT_EQ(it->second->remap_safe_event_, event);

  allocation.reset();
  ASSERT_EQ(cudaEventDestroy(event), cudaSuccess);
}

// --- P1: FreeImpl release path ---

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     FreeImplErasesRouteAndDelegates) {
  auto allocator = CreateAllocator();

  auto allocation = allocator->Allocate(256UL);
  ASSERT_NE(allocation, nullptr);
  auto* ptr = allocation->ptr();

  // Before free: route entry exists, sub-pool has 1 active block.
  EXPECT_EQ(allocator->active_allocations_.size(), 1UL);
  EXPECT_EQ(allocator->small_allocator_->allocated_blocks_.size(), 1UL);

  allocation.reset();

  // After free: route entry erased, sub-pool block freed.
  EXPECT_EQ(allocator->active_allocations_.size(), 0UL);
  EXPECT_EQ(allocator->small_allocator_->allocated_blocks_.size(), 0UL);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2, CrossPoolAllocFree) {
  auto allocator = CreateAllocator();

  // Allocate across both routed pools.
  auto small = allocator->Allocate(256UL);
  auto large = allocator->Allocate(2UL << 20);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(large, nullptr);
  EXPECT_EQ(allocator->active_allocations_.size(), 2UL);

  // Free in a different order than allocation.
  large.reset();
  EXPECT_EQ(allocator->active_allocations_.size(), 1UL);
  EXPECT_EQ(allocator->large_allocator_->allocated_blocks_.size(), 0UL);

  small.reset();
  EXPECT_EQ(allocator->active_allocations_.size(), 0UL);
  EXPECT_EQ(allocator->small_allocator_->allocated_blocks_.size(), 0UL);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     SetBlockRemapEventReturnsFalseForUnknownPtr) {
  auto allocator = CreateAllocator();

  EXPECT_FALSE(allocator->SetBlockRemapEvent(
      reinterpret_cast<void*>(0x1), nullptr, nullptr));
}

// --- P2: threshold boundary tests ---

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     SmallAllocationThresholdBoundary) {
  auto allocator = CreateAllocator();
  // CreateAllocator uses small_allocation_threshold = 2MB.

  // size = threshold - 1 → Small.
  auto just_below = allocator->Allocate((2UL << 20) - 1);
  ASSERT_NE(just_below, nullptr);
  EXPECT_EQ(allocator->active_allocations_[just_below->ptr()].allocator,
            allocator->small_allocator_.get());

  // size = threshold → Large (>= threshold goes to large).
  auto exact = allocator->Allocate(2UL << 20);
  ASSERT_NE(exact, nullptr);
  EXPECT_EQ(allocator->active_allocations_[exact->ptr()].allocator,
            allocator->large_allocator_.get());
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
