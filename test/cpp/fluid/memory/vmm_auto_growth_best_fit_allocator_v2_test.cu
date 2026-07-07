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
      phi::GPUPlace(), 2UL << 20, PoolType::kSmall);
}

__global__ void DelayedStoreKernel(uint8_t* ptr, uint64_t cycles) {
  uint64_t start = clock64();
  while (clock64() - start < cycles) {
  }
  ptr[0] = 1;
}

void ExpectBlockView(const BlockV2& block) { EXPECT_GT(block.size_, 0UL); }

}  // namespace

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitFreeBlockOnReuse) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

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
      ExpectBlockView(block);
    }
  }
  EXPECT_EQ(active_count, 1UL);
  EXPECT_EQ(free_count, 1UL);
  EXPECT_EQ(free_bytes, underlying->handle_size());
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ReuseSmallestSufficientFreeBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  // Layout after allocation:
  //   [ACTIVE 4MB] [ACTIVE 2MB separator] [ACTIVE 2MB small]
  // The separator prevents TryMerge from coalescing large and small on free.
  auto large = allocator.Allocate(underlying->handle_size() * 2);
  auto separator = allocator.Allocate(underlying->handle_size());
  auto small = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(large, nullptr);
  ASSERT_NE(separator, nullptr);
  ASSERT_NE(small, nullptr);

  auto* small_ptr = small->ptr();
  large.reset();
  small.reset();
  // Layout: [FREE 4MB] [ACTIVE 2MB separator] [FREE 2MB]
  // free_blocks_: {(2MB, ptr_small), (4MB, ptr_large)}

  auto reused = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(reused, nullptr);

  // lower_bound({2MB, nullptr}) picks the exact-fit 2MB free block over the
  // larger 4MB one.
  EXPECT_EQ(reused->ptr(), small_ptr);
  // Layout: [FREE 4MB] [ACTIVE 2MB separator] [ACTIVE 2MB reused]
  ASSERT_EQ(allocator.all_blocks_.size(), 3UL);
  size_t free_block_count = 0;
  for (const auto& block : allocator.all_blocks_) {
    if (block.type_ == BlockType::kFree) {
      ++free_block_count;
      EXPECT_EQ(block.size_, underlying->handle_size() * 2);
    }
  }
  EXPECT_EQ(free_block_count, 1UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ReturnedAllocationSizeMatchesRequest) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  auto allocation = allocator.Allocate(256UL);
  ASSERT_NE(allocation, nullptr);

  EXPECT_EQ(allocation->size(), 256UL);
  auto* alloc = static_cast<Allocation*>(allocation.get());
  EXPECT_EQ(alloc->ptr(), alloc->base_ptr());
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitGrowBlockAcrossTwoHandles) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  const size_t requested_size = underlying->handle_size() + 256UL;
  auto allocation = allocator.Allocate(requested_size);
  ASSERT_NE(allocation, nullptr);

  ASSERT_EQ(allocator.all_blocks_.size(), 2UL);
  auto it = allocator.all_blocks_.begin();
  ASSERT_EQ(it->type_, BlockType::kActive);
  EXPECT_EQ(it->size_, requested_size);
  ExpectBlockView(*it);

  ++it;
  ASSERT_EQ(it, std::prev(allocator.all_blocks_.end()));
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->size_, underlying->handle_size() - 256UL);
  ExpectBlockView(*it);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, MergeSplitFreeSlicesAsBlockView) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  auto allocation = allocator.Allocate(256UL);
  ASSERT_NE(allocation, nullptr);
  allocation.reset();

  ASSERT_EQ(allocator.all_blocks_.size(), 1UL);
  const auto& merged = allocator.all_blocks_.front();
  EXPECT_EQ(merged.type_, BlockType::kFree);
  EXPECT_EQ(merged.size_, underlying->handle_size());
  ExpectBlockView(merged);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, GrowExactHandleMultipleNoSplit) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  // Request exactly 1 handle_size — the bottom allocator returns the same
  // amount, so grow-split should produce NO remaining FREE block.
  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);

  EXPECT_EQ(allocator.all_blocks_.size(), 1UL);
  EXPECT_EQ(allocator.all_blocks_.front().type_, BlockType::kActive);
  EXPECT_EQ(allocator.all_blocks_.front().size_, underlying->handle_size());
  EXPECT_EQ(allocator.free_blocks_.size(), 0UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, AlignmentRoundsUpRequestedSize) {
  auto underlying = CreateUnderlyingAllocator();
  const size_t alignment = 512;
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, alignment, phi::GPUPlace(), PoolType::kSmall);

  // Request 100 bytes with alignment=512 → AlignedSize(100,512) = 512.
  auto allocation = allocator.Allocate(100);
  ASSERT_NE(allocation, nullptr);

  // The returned allocation size must be the aligned 512, not 100.
  EXPECT_EQ(allocation->size(), 512UL);

  // The ACTIVE block in all_blocks_ should also be 512.
  auto it = allocator.all_blocks_.begin();
  ASSERT_EQ(it->type_, BlockType::kActive);
  EXPECT_EQ(it->size_, 512UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ExactFitReuseNoSplit) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  // Allocate and free one handle — creates one FREE block of handle_size.
  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);
  auto* original_ptr = allocation->ptr();
  allocation.reset();
  ASSERT_EQ(allocator.free_blocks_.size(), 1UL);

  // Re-allocate exactly the same size — exact fit, no split needed.
  auto reused = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(reused, nullptr);
  EXPECT_EQ(reused->ptr(), original_ptr);

  // Only one block: ACTIVE, no FREE remainder.
  EXPECT_EQ(allocator.all_blocks_.size(), 1UL);
  EXPECT_EQ(allocator.all_blocks_.front().type_, BlockType::kActive);
  EXPECT_EQ(allocator.free_blocks_.size(), 0UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, AllocFreeCycleConsistency) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  // Perform several alloc/free cycles and verify invariants after each.
  for (int round = 0; round < 3; ++round) {
    auto a1 = allocator.Allocate(underlying->handle_size());
    auto a2 = allocator.Allocate(underlying->handle_size());
    ASSERT_NE(a1, nullptr);
    ASSERT_NE(a2, nullptr);
    size_t active_before_free = 0;
    for (const auto& block : allocator.all_blocks_) {
      if (block.type_ == BlockType::kActive) {
        ++active_before_free;
      }
    }
    EXPECT_EQ(active_before_free, 2UL);

    a1.reset();
    a2.reset();
    // After freeing all, adjacent blocks merge — should be exactly 1 FREE.
    EXPECT_EQ(allocator.free_blocks_.size(), 1UL);

    size_t total_free = 0;
    for (const auto& block : allocator.all_blocks_) {
      EXPECT_EQ(block.type_, BlockType::kFree);
      total_free += block.size_;
    }
    EXPECT_EQ(total_free, underlying->handle_size() * 2);
  }
}

TEST(VMMAutoGrowthBestFitAllocatorV2, FreeBlockTooSmallFallsBackToGrow) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  // Create a small free block (handle_size).
  auto small = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(small, nullptr);
  small.reset();
  ASSERT_EQ(allocator.free_blocks_.size(), 1UL);

  // Request 2*handle_size — free block is too small, must grow.
  auto large = allocator.Allocate(underlying->handle_size() * 2);
  ASSERT_NE(large, nullptr);

  // The old full underlying allocation is idle and may be released before grow.
  EXPECT_EQ(allocator.free_blocks_.size(), 0UL);

  // Verify total layout: only the new ACTIVE block remains.
  size_t active_count = 0;
  size_t free_count = 0;
  for (const auto& block : allocator.all_blocks_) {
    if (block.type_ == BlockType::kActive) {
      ++active_count;
      EXPECT_EQ(block.size_, underlying->handle_size() * 2);
    } else if (block.type_ == BlockType::kFree) {
      ++free_count;
    }
  }
  EXPECT_EQ(active_count, 1UL);
  EXPECT_EQ(free_count, 0UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     ReleaseIdleMiddleChunkLeavesReusableUnmappedFreeBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  auto first = allocator.Allocate(underlying->handle_size());
  auto middle = allocator.Allocate(underlying->handle_size());
  auto last = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(last, nullptr);

  auto* middle_ptr = middle->ptr();
  const size_t tail_after_allocs = underlying->tail_offset();
  middle.reset();

  EXPECT_EQ(allocator.Release(phi::GPUPlace()), underlying->handle_size());
  ASSERT_EQ(allocator.all_blocks_.size(), 3UL);
  auto it = allocator.all_blocks_.begin();
  EXPECT_TRUE(it->IsActive());
  ++it;
  ASSERT_TRUE(it->IsUnmappedFree());
  EXPECT_EQ(it->ptr_, middle_ptr);
  EXPECT_EQ(it->size_, underlying->handle_size());
  ++it;
  EXPECT_TRUE(it->IsActive());

  auto reused = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(reused, nullptr);
  EXPECT_EQ(reused->ptr(), middle_ptr);
  EXPECT_EQ(underlying->tail_offset(), tail_after_allocs);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ReleaseTailChunkRetreatsTailOffset) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);
  const size_t handle_size = underlying->handle_size();

  auto head = allocator.Allocate(handle_size);
  auto tail = allocator.Allocate(handle_size);
  ASSERT_NE(head, nullptr);
  ASSERT_NE(tail, nullptr);
  auto* expected_next_tail =
      reinterpret_cast<uint8_t*>(head->ptr()) + handle_size;

  tail.reset();
  EXPECT_EQ(allocator.Release(phi::GPUPlace()), handle_size);
  EXPECT_EQ(underlying->tail_offset(), handle_size);
  EXPECT_EQ(allocator.all_blocks_.size(), 1UL);
  EXPECT_EQ(allocator.unmapped_free_blocks_.size(), 0UL);

  auto grow = allocator.Allocate(handle_size * 2);
  ASSERT_NE(grow, nullptr);
  EXPECT_EQ(grow->ptr(), expected_next_tail);
  EXPECT_EQ(underlying->tail_offset(), handle_size * 3);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ReleaseWithNoIdleChunkReturnsZero) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  auto active = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(active, nullptr);
  const size_t tail_before_release = underlying->tail_offset();

  EXPECT_EQ(allocator.Release(phi::GPUPlace()), 0UL);
  EXPECT_EQ(underlying->tail_offset(), tail_before_release);
  ASSERT_EQ(allocator.all_blocks_.size(), 1UL);
  EXPECT_TRUE(allocator.all_blocks_.front().IsActive());
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     RangeOverlapsUnderlyingCoversRegistryCases) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);
  const size_t handle_size = underlying->handle_size();

  auto allocation = allocator.Allocate(handle_size);
  ASSERT_NE(allocation, nullptr);
  auto* ptr = reinterpret_cast<uint8_t*>(allocation->ptr());

  EXPECT_TRUE(allocator.RangeOverlapsUnderlying(ptr, handle_size));
  EXPECT_TRUE(allocator.RangeOverlapsUnderlying(ptr + handle_size / 2,
                                                handle_size / 2));
  EXPECT_FALSE(
      allocator.RangeOverlapsUnderlying(ptr + handle_size, handle_size));
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ReleasePredicatesRejectActiveAllocation) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);
  auto* ptr = reinterpret_cast<uint8_t*>(allocation->ptr());

  EXPECT_FALSE(allocator.IsRangeEntirelyFree(ptr, underlying->handle_size()));
  EXPECT_FALSE(
      allocator.CanReleaseIdleUnderlying(ptr, underlying->handle_size()));
  uint64_t released = 0;
  auto it = allocator.underlying_allocations_.begin();
  EXPECT_FALSE(allocator.TryReleaseIdleUnderlying(&it, &released));
  EXPECT_EQ(released, 0UL);
  EXPECT_EQ(allocator.FreeIdleChunks(), 0UL);

  allocation.reset();
  EXPECT_TRUE(allocator.IsRangeEntirelyFree(ptr, underlying->handle_size()));
}

TEST(VMMAutoGrowthBestFitAllocatorV2, FreeIndexHelpersIgnoreWrongBlockTypes) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  BlockV2 active = BlockV2::MakeMappedBlock(BlockType::kActive,
                                            reinterpret_cast<void*>(0x30000000),
                                            underlying->handle_size(),
                                            PoolType::kSmall);
  auto active_it = allocator.all_blocks_.insert(allocator.all_blocks_.end(),
                                                std::move(active));
  allocator.InsertFreeBlock(active_it);
  EXPECT_TRUE(allocator.free_blocks_.empty());

  BlockV2 mapped_free =
      BlockV2::MakeMappedBlock(BlockType::kFree,
                               reinterpret_cast<void*>(0x32000000),
                               underlying->handle_size(),
                               PoolType::kSmall);
  auto mapped_free_it = allocator.all_blocks_.insert(
      allocator.all_blocks_.end(), std::move(mapped_free));
  allocator.InsertUnmappedFreeBlock(mapped_free_it);
  EXPECT_TRUE(allocator.unmapped_free_blocks_.empty());

  allocator.TryMergeUnmappedFree(allocator.all_blocks_.end());
  EXPECT_TRUE(allocator.unmapped_free_blocks_.empty());
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     ReleaseAdjacentMiddleChunksMergeIntoSingleUnmappedFreeBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);
  const size_t handle_size = underlying->handle_size();

  auto first = allocator.Allocate(handle_size);
  auto middle_left = allocator.Allocate(handle_size);
  auto middle_right = allocator.Allocate(handle_size);
  auto last = allocator.Allocate(handle_size);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle_left, nullptr);
  ASSERT_NE(middle_right, nullptr);
  ASSERT_NE(last, nullptr);

  auto* middle_ptr = middle_left->ptr();
  middle_left.reset();
  middle_right.reset();

  EXPECT_EQ(allocator.Release(phi::GPUPlace()), handle_size * 2);
  ASSERT_EQ(allocator.all_blocks_.size(), 3UL);
  auto it = allocator.all_blocks_.begin();
  EXPECT_TRUE(it->IsActive());
  ++it;
  ASSERT_TRUE(it->IsUnmappedFree());
  EXPECT_EQ(it->ptr_, middle_ptr);
  EXPECT_EQ(it->size_, handle_size * 2);
  ++it;
  EXPECT_TRUE(it->IsActive());
  EXPECT_EQ(allocator.unmapped_free_blocks_.size(), 1UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     ReuseUnmappedFreeBlockWithMappedAndUnmappedRemainders) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);
  const size_t handle_size = underlying->handle_size();

  auto first = allocator.Allocate(handle_size);
  auto middle = allocator.Allocate(handle_size * 2);
  auto last = allocator.Allocate(handle_size);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(last, nullptr);

  auto* middle_ptr = middle->ptr();
  middle.reset();
  EXPECT_EQ(allocator.Release(phi::GPUPlace()), handle_size * 2);
  ASSERT_EQ(allocator.unmapped_free_blocks_.size(), 1UL);

  auto reused = allocator.Allocate(256UL);
  ASSERT_NE(reused, nullptr);
  EXPECT_EQ(reused->ptr(), middle_ptr);

  size_t active_count = 0;
  size_t mapped_free_bytes = 0;
  size_t unmapped_free_bytes = 0;
  for (const auto& block : allocator.all_blocks_) {
    if (block.IsActive()) {
      ++active_count;
    } else if (block.IsMappedFree()) {
      mapped_free_bytes += block.size_;
    } else if (block.IsUnmappedFree()) {
      unmapped_free_bytes += block.size_;
    }
  }
  EXPECT_EQ(active_count, 3UL);
  EXPECT_EQ(mapped_free_bytes, handle_size - 256UL);
  EXPECT_EQ(unmapped_free_bytes, handle_size);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     ReplaceRangeWithUnmappedFreeSplitsContainingFreeBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);
  const size_t handle_size = underlying->handle_size();

  auto allocation = allocator.Allocate(handle_size * 3);
  ASSERT_NE(allocation, nullptr);
  auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
  allocation.reset();
  ASSERT_EQ(allocator.all_blocks_.size(), 1UL);

  allocator.ReplaceRangeWithUnmappedFree(base + handle_size, handle_size);

  ASSERT_EQ(allocator.all_blocks_.size(), 3UL);
  auto it = allocator.all_blocks_.begin();
  EXPECT_TRUE(it->IsMappedFree());
  EXPECT_EQ(it->size_, handle_size);
  ++it;
  ASSERT_TRUE(it->IsUnmappedFree());
  EXPECT_EQ(it->ptr_, base + handle_size);
  EXPECT_EQ(it->size_, handle_size);
  ++it;
  EXPECT_TRUE(it->IsMappedFree());
  EXPECT_EQ(it->size_, handle_size);
  EXPECT_EQ(allocator.free_blocks_.size(), 2UL);
  EXPECT_EQ(allocator.unmapped_free_blocks_.size(), 1UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     ReplaceRangeWithUnmappedFreeKeepsLeftRemainder) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);
  const size_t handle_size = underlying->handle_size();

  auto allocation = allocator.Allocate(handle_size * 3);
  ASSERT_NE(allocation, nullptr);
  auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
  allocation.reset();

  allocator.ReplaceRangeWithUnmappedFree(base + handle_size, handle_size * 2);

  ASSERT_EQ(allocator.all_blocks_.size(), 2UL);
  auto it = allocator.all_blocks_.begin();
  EXPECT_TRUE(it->IsMappedFree());
  EXPECT_EQ(it->ptr_, base);
  EXPECT_EQ(it->size_, handle_size);
  ++it;
  EXPECT_TRUE(it->IsUnmappedFree());
  EXPECT_EQ(it->ptr_, base + handle_size);
  EXPECT_EQ(it->size_, handle_size * 2);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     ReplaceRangeWithUnmappedFreeKeepsRightRemainder) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);
  const size_t handle_size = underlying->handle_size();

  auto allocation = allocator.Allocate(handle_size * 3);
  ASSERT_NE(allocation, nullptr);
  auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
  allocation.reset();

  allocator.ReplaceRangeWithUnmappedFree(base, handle_size * 2);

  ASSERT_EQ(allocator.all_blocks_.size(), 2UL);
  auto it = allocator.all_blocks_.begin();
  EXPECT_TRUE(it->IsUnmappedFree());
  EXPECT_EQ(it->ptr_, base);
  EXPECT_EQ(it->size_, handle_size * 2);
  ++it;
  EXPECT_TRUE(it->IsMappedFree());
  EXPECT_EQ(it->ptr_, base + handle_size * 2);
  EXPECT_EQ(it->size_, handle_size);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, AdoptBackingBlockRejectsNullInput) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  EXPECT_THROW(allocator.AdoptBackingBlock(nullptr),
               common::enforce::EnforceNotMet);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ReleaseWaitsBeforeUnmappingBacking) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);
  auto* ptr = reinterpret_cast<uint8_t*>(allocation->ptr());

  DelayedStoreKernel<<<1, 1>>>(ptr, 20000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  allocation.reset();
  EXPECT_EQ(allocator.Release(phi::GPUPlace()), underlying->handle_size());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ThreeWayMerge) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kSmall);

  // Allocate 3 consecutive handle-sized blocks.
  auto a = allocator.Allocate(underlying->handle_size());
  auto b = allocator.Allocate(underlying->handle_size());
  auto c = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_NE(c, nullptr);
  ASSERT_EQ(allocator.all_blocks_.size(), 3UL);

  // Free first and last — creates 2 non-adjacent FREE blocks.
  a.reset();
  c.reset();
  EXPECT_EQ(allocator.free_blocks_.size(), 2UL);

  // Free middle — TryMerge merges prev+it (left), then merged+next (right)
  // into a single block spanning all 3 handles.
  b.reset();
  EXPECT_EQ(allocator.all_blocks_.size(), 1UL);
  EXPECT_EQ(allocator.free_blocks_.size(), 1UL);

  const auto& merged = allocator.all_blocks_.front();
  EXPECT_EQ(merged.type_, BlockType::kFree);
  EXPECT_EQ(merged.size_, underlying->handle_size() * 3);
  ExpectBlockView(merged);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
