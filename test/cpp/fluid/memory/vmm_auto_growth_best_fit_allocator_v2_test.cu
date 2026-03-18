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
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.h"
#undef private

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

std::shared_ptr<CUDAVirtualMemAllocatorV2> CreateUnderlyingAllocator() {
  return std::make_shared<CUDAVirtualMemAllocatorV2>(
      phi::GPUPlace(), 2UL << 20, PoolType::kTransient);
}

}  // namespace

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitFreeBlockOnReuse) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  auto large = allocator.Allocate(underlying->handle_size() * 2);
  ASSERT_NE(large, nullptr);
  large.reset();

  auto small = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(small, nullptr);

  ASSERT_EQ(allocator.all_blocks_.size(), 2UL);
  size_t active_count = 0;
  size_t free_count = 0;
  size_t free_bytes = 0;
  for (const auto& block : allocator.all_blocks_) {
    if (block.type_ == BlockType::kActive) {
      ++active_count;
      EXPECT_EQ(block.size_, underlying->handle_size());
    } else if (block.type_ == BlockType::kFree) {
      ++free_count;
      free_bytes += block.size_;
      EXPECT_EQ(block.parts_.size(), 1UL);
    }
  }
  EXPECT_EQ(active_count, 1UL);
  EXPECT_EQ(free_count, 1UL);
  EXPECT_EQ(free_bytes, underlying->handle_size());
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitGrowBlockOnFirstAllocation) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  // The bottom allocator rounds this grow to one full handle, but best-fit
  // should immediately split it into [ACTIVE requested_size] + [FREE remain].
  auto allocation = allocator.Allocate(256);
  ASSERT_NE(allocation, nullptr);

  ASSERT_EQ(allocator.all_blocks_.size(), 2UL);
  auto it = allocator.all_blocks_.begin();
  ASSERT_EQ(it->type_, BlockType::kActive);
  EXPECT_EQ(it->size_, 256UL);
  ASSERT_EQ(it->parts_.size(), 1UL);
  EXPECT_EQ(it->parts_[0].handle_rel_off, 0UL);
  EXPECT_EQ(it->parts_[0].len, 256UL);

  ++it;
  ASSERT_EQ(it, std::prev(allocator.all_blocks_.end()));
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->size_, underlying->handle_size() - 256UL);
  ASSERT_EQ(it->parts_.size(), 1UL);
  EXPECT_EQ(it->parts_[0].handle_rel_off, 256UL);
  EXPECT_EQ(it->parts_[0].len, underlying->handle_size() - 256UL);
  EXPECT_EQ(allocator.free_blocks_.size(), 1UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitGrowBlockAcrossTwoHandles) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  const size_t requested_size = underlying->handle_size() + 256UL;
  auto allocation = allocator.Allocate(requested_size);
  ASSERT_NE(allocation, nullptr);

  ASSERT_EQ(allocator.all_blocks_.size(), 2UL);
  auto it = allocator.all_blocks_.begin();
  ASSERT_EQ(it->type_, BlockType::kActive);
  EXPECT_EQ(it->size_, requested_size);
  ASSERT_EQ(it->parts_.size(), 2UL);
  EXPECT_EQ(it->parts_[0].handle_rel_off, 0UL);
  EXPECT_EQ(it->parts_[0].len, underlying->handle_size());
  EXPECT_EQ(it->parts_[1].handle_rel_off, 0UL);
  EXPECT_EQ(it->parts_[1].len, 256UL);

  ++it;
  ASSERT_EQ(it, std::prev(allocator.all_blocks_.end()));
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->size_, underlying->handle_size() - 256UL);
  ASSERT_EQ(it->parts_.size(), 1UL);
  EXPECT_EQ(it->parts_[0].handle_rel_off, 256UL);
  EXPECT_EQ(it->parts_[0].len, underlying->handle_size() - 256UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     MergeSplitFreeSlicesIntoSingleHandlePart) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  auto allocation = allocator.Allocate(256UL);
  ASSERT_NE(allocation, nullptr);
  allocation.reset();

  ASSERT_EQ(allocator.all_blocks_.size(), 1UL);
  const auto& merged = allocator.all_blocks_.front();
  EXPECT_EQ(merged.type_, BlockType::kFree);
  EXPECT_EQ(merged.size_, underlying->handle_size());
  ASSERT_EQ(merged.parts_.size(), 1UL);
  EXPECT_EQ(merged.parts_[0].handle_rel_off, 0UL);
  EXPECT_EQ(merged.parts_[0].len, underlying->handle_size());
}

TEST(VMMAutoGrowthBestFitAllocatorV2, MergeAdjacentFreeBlocks) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  auto whole = allocator.Allocate(underlying->handle_size() * 2);
  ASSERT_NE(whole, nullptr);
  whole.reset();

  auto first = allocator.Allocate(underlying->handle_size());
  auto second = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);

  first.reset();
  second.reset();

  ASSERT_EQ(allocator.all_blocks_.size(), 1UL);
  const auto& merged = allocator.all_blocks_.front();
  EXPECT_EQ(merged.type_, BlockType::kFree);
  EXPECT_EQ(merged.size_, underlying->handle_size() * 2);
  EXPECT_EQ(merged.parts_.size(), 2UL);
  EXPECT_EQ(allocator.free_blocks_.size(), 1UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitFreeBlockClearsRuntimeState) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);

  gpuEvent_t event = nullptr;
  ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            cudaSuccess);
  auto* ptr = allocation->ptr();
  ASSERT_TRUE(allocator.SetBlockRemapEvent(ptr, nullptr, event));
  auto active_it = allocator.allocated_blocks_.find(ptr);
  ASSERT_NE(active_it, allocator.allocated_blocks_.end());
  active_it->second->owning_stream_ = reinterpret_cast<gpuStream_t>(0x1);

  allocation.reset();

  auto reused = allocator.Allocate(256UL);
  ASSERT_NE(reused, nullptr);

  ASSERT_EQ(allocator.all_blocks_.size(), 2UL);
  size_t free_count = 0;
  for (const auto& block : allocator.all_blocks_) {
    if (block.type_ != BlockType::kFree) {
      continue;
    }
    ++free_count;
    EXPECT_EQ(block.owning_stream_, nullptr);
    EXPECT_EQ(block.last_use_stream_, nullptr);
    EXPECT_EQ(block.remap_safe_event_, nullptr);
  }
  EXPECT_EQ(free_count, 1UL);

  reused.reset();
  ASSERT_EQ(cudaEventDestroy(event), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SetBlockRemapEventStoresRuntimeState) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);

  gpuEvent_t event = nullptr;
  ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            cudaSuccess);
  auto* ptr = allocation->ptr();
  ASSERT_TRUE(allocator.SetBlockRemapEvent(ptr, nullptr, event));

  auto it = allocator.allocated_blocks_.find(ptr);
  ASSERT_NE(it, allocator.allocated_blocks_.end());
  EXPECT_EQ(it->second->last_use_stream_, nullptr);
  EXPECT_EQ(it->second->remap_safe_event_, event);

  allocation.reset();
  ASSERT_EQ(cudaEventDestroy(event), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SetBlockRemapEventRejectsUnknownPtr) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  EXPECT_FALSE(
      allocator.SetBlockRemapEvent(reinterpret_cast<void*>(0x1), nullptr,
                                   nullptr));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
