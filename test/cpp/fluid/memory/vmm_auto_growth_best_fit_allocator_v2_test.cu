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

TEST(VMMAutoGrowthBestFitAllocatorV2, ReuseSmallestSufficientFreeBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

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

TEST(VMMAutoGrowthBestFitAllocatorV2, ReturnedAllocationSizeMatchesRequest) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  auto allocation = allocator.Allocate(256UL);
  ASSERT_NE(allocation, nullptr);

  EXPECT_EQ(allocation->size(), 256UL);
  auto* alloc = static_cast<Allocation*>(allocation.get());
  EXPECT_EQ(alloc->ptr(), alloc->base_ptr());
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

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitGrowBlockStartsWithEmptyRemapState) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  auto allocation = allocator.Allocate(256UL);
  ASSERT_NE(allocation, nullptr);

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

TEST(VMMAutoGrowthBestFitAllocatorV2, NonAdjacentFreeBlocksDoNotMerge) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  auto first = allocator.Allocate(underlying->handle_size());
  auto middle = allocator.Allocate(underlying->handle_size());
  auto third = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(third, nullptr);

  first.reset();
  third.reset();

  ASSERT_EQ(allocator.all_blocks_.size(), 3UL);
  size_t free_count = 0;
  size_t active_count = 0;
  for (const auto& block : allocator.all_blocks_) {
    if (block.type_ == BlockType::kFree) {
      ++free_count;
      EXPECT_EQ(block.size_, underlying->handle_size());
      EXPECT_EQ(block.parts_.size(), 1UL);
    } else if (block.type_ == BlockType::kActive) {
      ++active_count;
      EXPECT_EQ(block.ptr_, middle->ptr());
    }
  }
  EXPECT_EQ(free_count, 2UL);
  EXPECT_EQ(active_count, 1UL);
  EXPECT_EQ(allocator.free_blocks_.size(), 2UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitFreeBlockInheritsRemapEvent) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);

  // Simulate StreamSafeCUDAAllocator injecting remap-safety metadata on free.
  gpuEvent_t event = nullptr;
  ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            cudaSuccess);
  auto* ptr = allocation->ptr();
  gpuStream_t fake_stream = reinterpret_cast<gpuStream_t>(0x1);
  ASSERT_TRUE(allocator.SetBlockRemapEvent(ptr, fake_stream, event));
  auto active_it = allocator.allocated_blocks_.find(ptr);
  ASSERT_NE(active_it, allocator.allocated_blocks_.end());
  active_it->second->owning_stream_ = fake_stream;

  allocation.reset();

  // Reuse with a smaller size triggers split. The remaining FREE block must
  // inherit last_use_stream_ / remap_safe_event_ from the original block so
  // that the Compactor knows the old kernel may still be touching this memory
  // (fast-GC same-stream reuse does not wait on the event).
  auto reused = allocator.Allocate(256UL);
  ASSERT_NE(reused, nullptr);

  ASSERT_EQ(allocator.all_blocks_.size(), 2UL);
  size_t free_count = 0;
  for (const auto& block : allocator.all_blocks_) {
    if (block.type_ != BlockType::kFree) {
      continue;
    }
    ++free_count;
    // owning_stream_ is cleared — nobody "owns" a free fragment.
    EXPECT_EQ(block.owning_stream_, nullptr);
    // last_use_stream_ and remap_safe_event_ are inherited — the old kernel
    // may still be accessing this region until the event completes.
    EXPECT_EQ(block.last_use_stream_, fake_stream);
    EXPECT_EQ(block.remap_safe_event_, event);
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

  EXPECT_FALSE(allocator.SetBlockRemapEvent(
      reinterpret_cast<void*>(0x1), nullptr, nullptr));
}

TEST(VMMAutoGrowthBestFitAllocatorV2, GrowExactHandleMultipleNoSplit) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  // Request exactly 1 handle_size — the bottom allocator returns the same
  // amount, so grow-split should produce NO remaining FREE block.
  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);

  EXPECT_EQ(allocator.all_blocks_.size(), 1UL);
  EXPECT_EQ(allocator.all_blocks_.front().type_, BlockType::kActive);
  EXPECT_EQ(allocator.all_blocks_.front().size_, underlying->handle_size());
  EXPECT_EQ(allocator.free_blocks_.size(), 0UL);
  EXPECT_EQ(allocator.allocated_blocks_.size(), 1UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, AlignmentRoundsUpRequestedSize) {
  auto underlying = CreateUnderlyingAllocator();
  const size_t alignment = 512;
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, alignment, phi::GPUPlace(), PoolType::kTransient);

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
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

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
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  // Perform several alloc/free cycles and verify invariants after each.
  for (int round = 0; round < 3; ++round) {
    auto a1 = allocator.Allocate(underlying->handle_size());
    auto a2 = allocator.Allocate(underlying->handle_size());
    ASSERT_NE(a1, nullptr);
    ASSERT_NE(a2, nullptr);
    EXPECT_EQ(allocator.allocated_blocks_.size(), 2UL);

    a1.reset();
    a2.reset();
    // After freeing all, adjacent blocks merge — should be exactly 1 FREE.
    EXPECT_EQ(allocator.allocated_blocks_.size(), 0UL);
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
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

  // Create a small free block (handle_size).
  auto small = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(small, nullptr);
  small.reset();
  ASSERT_EQ(allocator.free_blocks_.size(), 1UL);

  // Request 2*handle_size — free block is too small, must grow.
  auto large = allocator.Allocate(underlying->handle_size() * 2);
  ASSERT_NE(large, nullptr);

  // The old free block should still exist, and a new block was grown.
  EXPECT_EQ(allocator.free_blocks_.size(), 1UL);
  EXPECT_EQ(allocator.allocated_blocks_.size(), 1UL);

  // Verify total layout: 1 FREE (old) + 1 ACTIVE (new large).
  size_t active_count = 0;
  size_t free_count = 0;
  for (const auto& block : allocator.all_blocks_) {
    if (block.type_ == BlockType::kActive) {
      ++active_count;
      EXPECT_EQ(block.size_, underlying->handle_size() * 2);
    } else if (block.type_ == BlockType::kFree) {
      ++free_count;
      EXPECT_EQ(block.size_, underlying->handle_size());
    }
  }
  EXPECT_EQ(active_count, 1UL);
  EXPECT_EQ(free_count, 1UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ThreeWayMerge) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kTransient);

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
  EXPECT_EQ(merged.parts_.size(), 3UL);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
