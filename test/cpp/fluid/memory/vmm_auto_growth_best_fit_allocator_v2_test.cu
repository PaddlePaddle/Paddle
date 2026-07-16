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

#include <stdexcept>

#include "glog/logging.h"
#include "gtest/gtest.h"

#define private public
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.h"
#undef private

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/free_block_remap_compactor.h"
#include "paddle/phi/core/memory/allocation/remap_transaction.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

std::shared_ptr<CUDAVirtualMemAllocatorV2> CreateUnderlyingAllocator() {
  return std::make_shared<CUDAVirtualMemAllocatorV2>(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);
}

__global__ void BusyWaitKernel(uint64_t cycles) {
  uint64_t start = clock64();
  while (clock64() - start < cycles) {
  }
}

__global__ void DelayedStoreKernel(uint8_t* ptr, uint64_t cycles) {
  uint64_t start = clock64();
  while (clock64() - start < cycles) {
  }
  ptr[0] = 1;
}

size_t CountBlocksOfType(const VMMAutoGrowthBestFitAllocatorV2& allocator,
                         BlockType type) {
  size_t count = 0;
  for (const auto& block : allocator.all_blocks()) {
    if (block.type_ == type) {
      ++count;
    }
  }
  return count;
}

const BlockV2* FindBlockByPtr(const VMMAutoGrowthBestFitAllocatorV2& allocator,
                              void* ptr) {
  for (const auto& block : allocator.all_blocks()) {
    if (block.ptr_ == ptr) {
      return &block;
    }
  }
  return nullptr;
}

void ExpectIndexedFreeStats(VMMAutoGrowthBestFitAllocatorV2* allocator,
                            size_t total_free,
                            size_t max_free) {
  size_t actual_total_free = 0;
  size_t actual_max_free = 0;
  allocator->GetFreeBlockStats(&actual_total_free, &actual_max_free);
  EXPECT_EQ(actual_total_free, total_free);
  EXPECT_EQ(actual_max_free, max_free);
}

void ExpectBlockView(const BlockV2& block) { EXPECT_GT(block.size_, 0UL); }

class ScopedVLogLevel {
 public:
  explicit ScopedVLogLevel(int level) : old_level_(FLAGS_v) { FLAGS_v = level; }
  ~ScopedVLogLevel() { FLAGS_v = old_level_; }

 private:
  int old_level_;
};

void MarkRemapSafeForTest(phi::Allocation* allocation) {
  auto* remap_allocation = dynamic_cast<VMMRemapEventAllocation*>(allocation);
  ASSERT_NE(remap_allocation, nullptr);
  ASSERT_TRUE(remap_allocation->SetVMMRemapEvent(cudaStreamPerThread, nullptr));
}

}  // namespace

TEST(VMMAutoGrowthBestFitAllocatorV2, ReuseSmallestSufficientFreeBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

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
  ASSERT_EQ(allocator.all_blocks().size(), 3UL);
  size_t free_block_count = 0;
  for (const auto& block : allocator.all_blocks()) {
    if (block.type_ == BlockType::kFree) {
      ++free_block_count;
      EXPECT_EQ(block.size_, underlying->handle_size() * 2);
    }
  }
  EXPECT_EQ(free_block_count, 1UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, RejectsInvalidInternalOperations) {
  ScopedVLogLevel vlog_guard(6);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);
  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);

  EXPECT_FALSE(allocator.SetBlockRemapEvent(
      allocator.all_blocks_.end(), cudaStreamPerThread, nullptr));
  ASSERT_TRUE(allocator.SetBlockRemapEvent(
      allocator.all_blocks_.begin(), cudaStreamPerThread, nullptr));
  auto snapshot = allocator.SnapshotAllBlocks();
  ASSERT_EQ(snapshot.size(), 1UL);
  EXPECT_EQ(snapshot.front().ptr_, allocation->ptr());
  ASSERT_NE(allocator.underlying_allocations_.begin(),
            allocator.underlying_allocations_.end());
  const auto& underlying_allocation =
      *allocator.underlying_allocations_.begin();
  EXPECT_FALSE(allocator.IsRemapDestinationAllocation(underlying_allocation));

  uint64_t released = 0;
  auto underlying_it = allocator.underlying_allocations_.begin();
  EXPECT_FALSE(
      allocator.TryReleaseUnderlyingAllocation(&underlying_it, &released));
  EXPECT_EQ(released, 0UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitGrowBlockAcrossTwoHandles) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t requested_size = underlying->handle_size() + 256UL;
  auto allocation = allocator.Allocate(requested_size);
  ASSERT_NE(allocation, nullptr);

  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  auto it = allocator.all_blocks().begin();
  ASSERT_EQ(it->type_, BlockType::kActive);
  EXPECT_EQ(it->size_, requested_size);
  ExpectBlockView(*it);

  ++it;
  ASSERT_EQ(it, std::prev(allocator.all_blocks().end()));
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->size_, underlying->handle_size() - 256UL);
  ExpectBlockView(*it);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, MergeSplitFreeSlicesAsBlockView) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(256UL);
  ASSERT_NE(allocation, nullptr);
  allocation.reset();

  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  const auto& merged = allocator.all_blocks().front();
  EXPECT_EQ(merged.type_, BlockType::kFree);
  EXPECT_EQ(merged.size_, underlying->handle_size());
  ExpectBlockView(merged);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SplitFreeBlockAfterRemapEvent) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);

  // Simulate StreamSafeCUDAAllocator injecting remap-safety metadata on free.
  gpuEvent_t event = nullptr;
  ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            cudaSuccess);
  auto guard = std::make_shared<CUDAEventGuard>(event);
  auto* ptr = allocation->ptr();
  gpuStream_t fake_stream = reinterpret_cast<gpuStream_t>(0x1);
  ASSERT_TRUE(allocator.SetBlockRemapEvent(ptr, fake_stream, guard));

  allocation.reset();

  // Reuse with a smaller size triggers split. Pending-event state is now
  // tracked by BackingMap rather than handle metadata.
  auto reused = allocator.Allocate(256UL);
  ASSERT_NE(reused, nullptr);

  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  size_t free_count = 0;
  for (const auto& block : allocator.all_blocks()) {
    if (block.type_ != BlockType::kFree) {
      continue;
    }
    ++free_count;
    // owning_stream_ is cleared; nobody "owns" a free fragment.
    EXPECT_EQ(block.owning_stream_, nullptr);
    ExpectBlockView(block);
  }
  EXPECT_EQ(free_count, 1UL);

  reused.reset();
}

TEST(VMMAutoGrowthBestFitAllocatorV2, MergeFreeBlocksWithDifferentStreams) {
  ScopedVLogLevel vlog_guard(4);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->handle_size());
  auto second = allocator.Allocate(underlying->handle_size());
  auto tail_guard = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(tail_guard, nullptr);

  gpuStream_t first_stream;
  gpuStream_t second_stream;
  ASSERT_EQ(cudaStreamCreate(&first_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&second_stream), cudaSuccess);
  BusyWaitKernel<<<1, 1, 0, first_stream>>>(500000000ULL);
  BusyWaitKernel<<<1, 1, 0, second_stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  auto* first_remap = dynamic_cast<VMMRemapEventAllocation*>(first.get());
  auto* second_remap = dynamic_cast<VMMRemapEventAllocation*>(second.get());
  ASSERT_NE(first_remap, nullptr);
  ASSERT_NE(second_remap, nullptr);
  ASSERT_TRUE(first_remap->SetVMMRemapEvent(first_stream, nullptr));
  ASSERT_TRUE(second_remap->SetVMMRemapEvent(second_stream, nullptr));

  first.reset();
  second.reset();

  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  const auto& merged = allocator.all_blocks().front();
  EXPECT_EQ(merged.type_, BlockType::kFree);
  EXPECT_EQ(merged.size_, 2UL * underlying->handle_size());
  EXPECT_EQ(merged.remap_pending_states_.size(), 1UL);

  EXPECT_EQ(allocator.Compact(phi::GPUPlace()), 0UL);

  ASSERT_EQ(cudaStreamSynchronize(first_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(second_stream), cudaSuccess);
  EXPECT_EQ(allocator.Compact(phi::GPUPlace()),
            2UL * underlying->handle_size());

  ASSERT_EQ(cudaStreamDestroy(first_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(second_stream), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, MergeCopiesRemapSafetyFromNextBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->handle_size());
  auto second = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);

  gpuStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  BusyWaitKernel<<<1, 1, 0, stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  auto* remap_allocation = dynamic_cast<VMMRemapEventAllocation*>(second.get());
  ASSERT_NE(remap_allocation, nullptr);
  ASSERT_TRUE(remap_allocation->SetVMMRemapEvent(stream, nullptr));
  first.reset();
  second.reset();

  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  const auto& merged = allocator.all_blocks().front();
  EXPECT_EQ(merged.size_, 2UL * underlying->handle_size());
  EXPECT_TRUE(merged.owning_stream_ != nullptr ||
              !merged.remap_pending_states_.empty());
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, RestoreUnmappedRangeToMappedFreeBlock) {
  ScopedVLogLevel vlog_guard(3);
  auto underlying = CreateUnderlyingAllocator();
  const size_t handle_size = underlying->handle_size();
  RemapTransaction transaction(underlying.get(), handle_size);
  RemapTransaction::BlockList blocks;
  auto base = underlying->virtual_mem_base();
  blocks.push_back(BlockV2::MakeMappedBlock(BlockType::kActive,
                                            reinterpret_cast<void*>(base),
                                            handle_size,
                                            PoolType::kLarge));
  blocks.push_back(BlockV2::MakeUnmappedFreeBlock(
      reinterpret_cast<void*>(base + handle_size),
      3UL * handle_size,
      PoolType::kLarge));

  EXPECT_TRUE(transaction.RestoreRangeAsMappedFree(
      &blocks, base + 2UL * handle_size, handle_size));
  ASSERT_EQ(blocks.size(), 4UL);
  auto it = blocks.begin();
  EXPECT_TRUE(it->IsActive());
  ++it;
  EXPECT_TRUE(it->IsUnmappedFree());
  EXPECT_EQ(it->size(), handle_size);
  ++it;
  EXPECT_TRUE(it->IsMappedFree());
  EXPECT_EQ(it->size(), handle_size);
  ++it;
  EXPECT_TRUE(it->IsUnmappedFree());
  EXPECT_EQ(it->size(), handle_size);

  RemapTransaction::BlockList exceeds_blocks;
  exceeds_blocks.push_back(BlockV2::MakeUnmappedFreeBlock(
      reinterpret_cast<void*>(base), handle_size, PoolType::kLarge));
  EXPECT_FALSE(transaction.RestoreRangeAsMappedFree(
      &exceeds_blocks, base, 2UL * handle_size));

  RemapTransaction::BlockList missing_blocks;
  missing_blocks.push_back(
      BlockV2::MakeMappedBlock(BlockType::kActive,
                               reinterpret_cast<void*>(base),
                               handle_size,
                               PoolType::kLarge));
  EXPECT_FALSE(transaction.RestoreRangeAsMappedFree(
      &missing_blocks, base + handle_size, handle_size));
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ForceReleasedRemapSourceStaysUnmapped) {
  auto underlying = CreateUnderlyingAllocator();
  const size_t handle_size = underlying->handle_size();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(handle_size);
  ASSERT_NE(allocation, nullptr);
  const auto source_va = reinterpret_cast<VMMDevicePtr>(allocation->ptr());
  allocation.reset();

  auto source_pages =
      underlying->CollectMappedPages({{source_va, handle_size}}, handle_size);
  ASSERT_EQ(source_pages.size(), 1UL);
  ASSERT_NE(source_pages[0].meta, nullptr);
  ASSERT_TRUE(underlying->UnmapMappedRangeForRemap(source_va, 1));
  source_pages[0].meta->MarkOwnedByRemapDestination();

  // Make source restoration fail after cuMemMap. The rollback must keep the
  // released source VA out of the mapped-free best-fit index.
  underlying->access_desc_.clear();
  RemapTransaction transaction(underlying.get(), handle_size);
  transaction.RestoreRemappedSourcesToFreeBlocks(&allocator.all_blocks_,
                                                 source_pages);
  allocator.RebuildFreeBlockIndex();

  ASSERT_EQ(allocator.all_blocks_.size(), 1UL);
  EXPECT_TRUE(allocator.all_blocks_.front().IsUnmappedFree());
  EXPECT_TRUE(underlying->IsRangeUnmapped(source_va, handle_size));
  ExpectIndexedFreeStats(&allocator, 0UL, 0UL);
  EXPECT_EQ(allocator.unmapped_free_blocks_.size(), 1UL);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, FreeBlockTooSmallFallsBackToGrow) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  // Create a small free block (handle_size).
  auto small = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(small, nullptr);
  small.reset();
  ExpectIndexedFreeStats(
      &allocator, underlying->handle_size(), underlying->handle_size());

  // Request 2*handle_size: free block is too small, must grow.
  auto large = allocator.Allocate(underlying->handle_size() * 2);
  ASSERT_NE(large, nullptr);

  // The old tail free block is used as the prefix of the new allocation, and
  // only the missing suffix is grown from the bottom allocator.
  ExpectIndexedFreeStats(&allocator, 0UL, 0UL);
  EXPECT_EQ(CountBlocksOfType(allocator, BlockType::kActive), 1UL);

  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  EXPECT_EQ(allocator.all_blocks().front().type_, BlockType::kActive);
  EXPECT_EQ(allocator.all_blocks().front().size_,
            underlying->handle_size() * 2);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ThreeWayMerge) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  // Allocate 3 consecutive handle-sized blocks.
  auto a = allocator.Allocate(underlying->handle_size());
  auto b = allocator.Allocate(underlying->handle_size());
  auto c = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_NE(c, nullptr);
  ASSERT_EQ(allocator.all_blocks().size(), 3UL);

  // Free first and last: creates 2 non-adjacent FREE blocks.
  a.reset();
  c.reset();
  ExpectIndexedFreeStats(
      &allocator, underlying->handle_size() * 2, underlying->handle_size());

  // Free middle: TryMerge merges prev+it (left), then merged+next (right)
  // into a single block spanning all 3 handles.
  b.reset();
  EXPECT_EQ(allocator.all_blocks().size(), 1UL);
  ExpectIndexedFreeStats(
      &allocator, underlying->handle_size() * 3, underlying->handle_size() * 3);

  const auto& merged = allocator.all_blocks().front();
  EXPECT_EQ(merged.type_, BlockType::kFree);
  EXPECT_EQ(merged.size_, underlying->handle_size() * 3);
  ExpectBlockView(merged);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactRemapsWholeFreeHandleToTail) {
  ScopedVLogLevel vlog_guard(10);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->handle_size());
  auto middle = allocator.Allocate(underlying->handle_size());
  auto last = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(last, nullptr);

  auto* first_ptr = first->ptr();
  auto* middle_ptr = middle->ptr();
  auto* last_ptr = last->ptr();
  std::vector<std::pair<VMMDevicePtr, size_t>> middle_range = {
      {reinterpret_cast<VMMDevicePtr>(middle_ptr), underlying->handle_size()}};
  const auto middle_pages =
      underlying->CollectMappedPages(middle_range, underlying->handle_size());
  ASSERT_EQ(middle_pages.size(), 1UL);

  MarkRemapSafeForTest(middle.get());
  middle.reset();
  const size_t remapped = allocator.Compact(phi::GPUPlace());
  EXPECT_EQ(remapped, underlying->handle_size());

  ASSERT_EQ(allocator.all_blocks().size(), 4UL);
  auto it = allocator.all_blocks().begin();
  ASSERT_EQ(it->type_, BlockType::kActive);
  EXPECT_EQ(it->ptr_, first_ptr);
  ++it;
  ASSERT_EQ(it->type_, BlockType::kUnmappedFree);
  EXPECT_EQ(it->ptr_, middle_ptr);
  EXPECT_EQ(it->size_, underlying->handle_size());
  ++it;
  ASSERT_EQ(it->type_, BlockType::kActive);
  EXPECT_EQ(it->ptr_, last_ptr);
  ++it;
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->size_, underlying->handle_size());
  ExpectBlockView(*it);
  std::vector<std::pair<VMMDevicePtr, size_t>> tail_range = {
      {reinterpret_cast<VMMDevicePtr>(it->ptr_), underlying->handle_size()}};
  const auto tail_pages =
      underlying->CollectMappedPages(tail_range, underlying->handle_size());
  ASSERT_EQ(tail_pages.size(), 1UL);
  EXPECT_EQ(tail_pages[0].handle, middle_pages[0].handle);
  ExpectIndexedFreeStats(
      &allocator, underlying->handle_size(), underlying->handle_size());
}

TEST(VMMAutoGrowthBestFitAllocatorV2, SkipStaleUnmappedFreeRange) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->handle_size());
  auto middle = allocator.Allocate(underlying->handle_size());
  auto last = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(last, nullptr);

  auto* middle_ptr = middle->ptr();
  MarkRemapSafeForTest(middle.get());
  middle.reset();

  ASSERT_EQ(allocator.Compact(phi::GPUPlace()), underlying->handle_size());
  ExpectIndexedFreeStats(
      &allocator, underlying->handle_size(), underlying->handle_size());

  auto tail_reuse = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(tail_reuse, nullptr);
  EXPECT_NE(tail_reuse->ptr(), middle_ptr);
  ExpectIndexedFreeStats(&allocator, 0UL, 0UL);

  const size_t tail_before_unmapped_reuse = underlying->tail_offset();
  auto unmapped_reuse = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(unmapped_reuse, nullptr);
  EXPECT_NE(unmapped_reuse->ptr(), middle_ptr);
  EXPECT_GT(underlying->tail_offset(), tail_before_unmapped_reuse);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ReleaseMiddleChunk) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->handle_size());
  auto middle = allocator.Allocate(underlying->handle_size());
  auto last = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(last, nullptr);

  auto* middle_ptr = middle->ptr();
  const size_t tail_after_allocs = underlying->tail_offset();
  middle.reset();

  const uint64_t released = allocator.Release(phi::GPUPlace());
  EXPECT_EQ(released, underlying->handle_size());
  ASSERT_EQ(allocator.all_blocks().size(), 3UL);
  auto it = allocator.all_blocks().begin();
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
  EXPECT_EQ(allocator.all_blocks().size(), 1UL);
  EXPECT_FALSE(allocator.all_blocks().back().IsUnmappedFree());

  auto grow = allocator.Allocate(handle_size * 2);
  ASSERT_NE(grow, nullptr);
  EXPECT_EQ(grow->ptr(), expected_next_tail);
  EXPECT_EQ(underlying->tail_offset(), handle_size * 3);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ReleaseWithNoIdleChunkReturnsZero) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto active = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(active, nullptr);
  const size_t tail_before_release = underlying->tail_offset();

  EXPECT_EQ(allocator.Release(phi::GPUPlace()), 0UL);
  EXPECT_EQ(underlying->tail_offset(), tail_before_release);
  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  EXPECT_TRUE(allocator.all_blocks().front().IsActive());
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ReuseUnmappedFreeBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);
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
  for (const auto& block : allocator.all_blocks()) {
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

TEST(VMMAutoGrowthBestFitAllocatorV2, ReplaceRangeSplitsFreeBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);
  const size_t handle_size = underlying->handle_size();

  auto allocation = allocator.Allocate(handle_size * 3);
  ASSERT_NE(allocation, nullptr);
  auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
  allocation.reset();
  ASSERT_EQ(allocator.all_blocks().size(), 1UL);

  allocator.ReplaceRangeWithUnmappedFree(base + handle_size, handle_size);

  ASSERT_EQ(allocator.all_blocks().size(), 3UL);
  auto it = allocator.all_blocks().begin();
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

TEST(VMMAutoGrowthBestFitAllocatorV2, ReplaceRangeTrimsBoundaryBlocks) {
  const size_t handle_size = 2UL << 20;

  {
    auto underlying = CreateUnderlyingAllocator();
    VMMAutoGrowthBestFitAllocatorV2 allocator(
        underlying, 256, phi::GPUPlace(), PoolType::kLarge);

    auto allocation = allocator.Allocate(handle_size * 3);
    ASSERT_NE(allocation, nullptr);
    auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
    allocation.reset();

    allocator.ReplaceRangeWithUnmappedFree(base + handle_size, handle_size * 2);

    ASSERT_EQ(allocator.all_blocks().size(), 2UL);
    auto it = allocator.all_blocks().begin();
    EXPECT_TRUE(it->IsMappedFree());
    EXPECT_EQ(it->ptr_, base);
    EXPECT_EQ(it->size_, handle_size);
    ++it;
    EXPECT_TRUE(it->IsUnmappedFree());
    EXPECT_EQ(it->ptr_, base + handle_size);
    EXPECT_EQ(it->size_, handle_size * 2);
  }

  {
    auto underlying = CreateUnderlyingAllocator();
    VMMAutoGrowthBestFitAllocatorV2 allocator(
        underlying, 256, phi::GPUPlace(), PoolType::kLarge);

    auto allocation = allocator.Allocate(handle_size * 3);
    ASSERT_NE(allocation, nullptr);
    auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
    allocation.reset();

    allocator.ReplaceRangeWithUnmappedFree(base, handle_size * 2);

    ASSERT_EQ(allocator.all_blocks().size(), 2UL);
    auto it = allocator.all_blocks().begin();
    EXPECT_TRUE(it->IsUnmappedFree());
    EXPECT_EQ(it->ptr_, base);
    EXPECT_EQ(it->size_, handle_size * 2);
    ++it;
    EXPECT_TRUE(it->IsMappedFree());
    EXPECT_EQ(it->ptr_, base + handle_size * 2);
    EXPECT_EQ(it->size_, handle_size);
  }
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ReleaseWaitsBeforeUnmappingBacking) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);
  auto* ptr = reinterpret_cast<uint8_t*>(allocation->ptr());

  DelayedStoreKernel<<<1, 1>>>(ptr, 20000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  allocation.reset();
  EXPECT_EQ(allocator.Release(phi::GPUPlace()), underlying->handle_size());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, IPCExportKeepsReuse) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->handle_size() * 2);
  ASSERT_NE(allocation, nullptr);
  auto* ptr = allocation->ptr();
  const size_t tail_after_first_alloc = underlying->tail_offset();
  auto* tensor_ptr = reinterpret_cast<uint8_t*>(ptr) + 128UL;
  const size_t tensor_size = underlying->handle_size() - 128UL + 2048UL;

  std::vector<BlockPart> parts;
  ASSERT_TRUE(allocator.CollectTensorParts(tensor_ptr, tensor_size, &parts));
  ASSERT_EQ(parts.size(), 2UL);
  EXPECT_EQ(parts[0].chunk_rel_off, 128UL);
  EXPECT_EQ(parts[0].len, underlying->handle_size() - 128UL);
  EXPECT_EQ(parts[1].chunk_rel_off, 0UL);
  EXPECT_EQ(parts[1].len, 2048UL);

  EXPECT_EQ(parts[0].chunk->base, reinterpret_cast<VMMDevicePtr>(ptr));
  EXPECT_EQ(parts[0].chunk->size, underlying->handle_size());
  EXPECT_EQ(parts[1].chunk->base,
            reinterpret_cast<VMMDevicePtr>(ptr) + underlying->handle_size());
  std::vector<std::pair<VMMDevicePtr, size_t>> exported_ranges = {
      {reinterpret_cast<VMMDevicePtr>(ptr), underlying->handle_size() * 2}};
  EXPECT_TRUE(
      underlying
          ->CollectMappedPages(exported_ranges, underlying->handle_size() * 2)
          .empty());

  allocation.reset();
  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  EXPECT_TRUE(allocator.all_blocks().front().IsFree());
  EXPECT_TRUE(underlying->HasIPCExportedRange(
      reinterpret_cast<VMMDevicePtr>(ptr), underlying->handle_size() * 2));
  ExpectIndexedFreeStats(
      &allocator, underlying->handle_size() * 2, underlying->handle_size() * 2);

  auto released = allocator.Release(phi::GPUPlace());
  EXPECT_EQ(released, 0UL);
  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  EXPECT_TRUE(underlying->HasIPCExportedRange(
      reinterpret_cast<VMMDevicePtr>(ptr), underlying->handle_size() * 2));

  auto remapped = allocator.RemapForAllocation(phi::GPUPlace(),
                                               underlying->handle_size() * 2);
  EXPECT_EQ(remapped, 0UL);

  auto next = allocator.Allocate(underlying->handle_size() * 2);
  ASSERT_NE(next, nullptr);
  EXPECT_EQ(next->ptr(), ptr);
  EXPECT_EQ(underlying->tail_offset(), tail_after_first_alloc);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CollectTensorPartsRejectsFreeRange) {
  ScopedVLogLevel vlog_guard(8);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);
  auto* ptr = allocation->ptr();
  allocation.reset();

  std::vector<BlockPart> parts;
  EXPECT_FALSE(
      allocator.CollectTensorParts(ptr, underlying->handle_size(), &parts));
  EXPECT_TRUE(parts.empty());
}

TEST(VMMAutoGrowthBestFitAllocatorV2, IPCPinAllowsNeighborRelease) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto exported = allocator.Allocate(underlying->handle_size());
  auto regular = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(exported, nullptr);
  ASSERT_NE(regular, nullptr);
  auto* exported_ptr = exported->ptr();

  std::vector<BlockPart> parts;
  ASSERT_TRUE(allocator.CollectTensorParts(
      exported_ptr, underlying->handle_size(), &parts));

  regular.reset();
  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  ExpectIndexedFreeStats(
      &allocator, underlying->handle_size(), underlying->handle_size());

  exported.reset();
  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  EXPECT_TRUE(allocator.all_blocks().front().IsFree());
  EXPECT_TRUE(underlying->HasIPCExportedRange(
      reinterpret_cast<VMMDevicePtr>(exported_ptr), underlying->handle_size()));
  EXPECT_EQ(allocator.all_blocks().front().size_,
            underlying->handle_size() * 2);
  ExpectIndexedFreeStats(
      &allocator, underlying->handle_size() * 2, underlying->handle_size() * 2);

  auto released = allocator.Release(phi::GPUPlace());
  EXPECT_EQ(released, underlying->handle_size());
  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  auto block_it = allocator.all_blocks().begin();
  ASSERT_TRUE(block_it->IsFree());
  EXPECT_EQ(block_it->ptr_, exported_ptr);
  EXPECT_EQ(block_it->size_, underlying->handle_size());
  EXPECT_EQ(underlying->tail_offset(), underlying->handle_size());
  EXPECT_TRUE(underlying->HasIPCExportedRange(
      reinterpret_cast<VMMDevicePtr>(exported_ptr), underlying->handle_size()));

  auto next = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(next, nullptr);
  EXPECT_EQ(next->ptr(), exported_ptr);
  EXPECT_EQ(underlying->tail_offset(), underlying->handle_size());
}

TEST(VMMAutoGrowthBestFitAllocatorV2, IPCPinAllowsNeighborCompact) {
  ScopedVLogLevel vlog_guard(4);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto exported = allocator.Allocate(underlying->handle_size());
  auto regular = allocator.Allocate(underlying->handle_size());
  auto anchor = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(exported, nullptr);
  ASSERT_NE(regular, nullptr);
  ASSERT_NE(anchor, nullptr);
  auto* exported_ptr = exported->ptr();
  auto* regular_ptr = regular->ptr();

  std::vector<BlockPart> parts;
  ASSERT_TRUE(allocator.CollectTensorParts(
      exported_ptr, underlying->handle_size(), &parts));

  MarkRemapSafeForTest(regular.get());
  MarkRemapSafeForTest(exported.get());
  regular.reset();
  exported.reset();
  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  EXPECT_TRUE(allocator.all_blocks().front().IsFree());
  EXPECT_TRUE(underlying->HasIPCExportedRange(
      reinterpret_cast<VMMDevicePtr>(exported_ptr), underlying->handle_size()));
  EXPECT_EQ(allocator.all_blocks().front().size_,
            underlying->handle_size() * 2);
  ExpectIndexedFreeStats(
      &allocator, underlying->handle_size() * 2, underlying->handle_size() * 2);

  auto remapped = allocator.RemapForAllocation(phi::GPUPlace(),
                                               underlying->handle_size() * 3);
  EXPECT_EQ(remapped, underlying->handle_size());

  const auto* exported_block = FindBlockByPtr(allocator, exported_ptr);
  ASSERT_NE(exported_block, nullptr);
  EXPECT_TRUE(exported_block->IsFree());
  EXPECT_EQ(exported_block->size_, underlying->handle_size());
  EXPECT_TRUE(underlying->HasIPCExportedRange(
      reinterpret_cast<VMMDevicePtr>(exported_ptr), underlying->handle_size()));

  const auto* regular_block = FindBlockByPtr(allocator, regular_ptr);
  ASSERT_NE(regular_block, nullptr);
  EXPECT_TRUE(regular_block->IsUnmappedFree());
  EXPECT_EQ(regular_block->size_, underlying->handle_size());
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactMergesUnmappedSources) {
  ScopedVLogLevel vlog_guard(4);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->handle_size());
  auto second = allocator.Allocate(underlying->handle_size());
  auto anchor = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(anchor, nullptr);
  auto* first_ptr = first->ptr();
  auto* second_ptr = second->ptr();

  MarkRemapSafeForTest(first.get());
  first.reset();
  auto first_compacted = allocator.Compact(phi::GPUPlace());
  EXPECT_EQ(first_compacted, underlying->handle_size());

  auto first_block = allocator.all_blocks().begin();
  ASSERT_NE(first_block, allocator.all_blocks().end());
  ASSERT_TRUE(first_block->IsUnmappedFree());
  EXPECT_EQ(first_block->ptr_, first_ptr);
  EXPECT_EQ(first_block->size_, underlying->handle_size());

  MarkRemapSafeForTest(second.get());
  second.reset();
  auto second_compacted = allocator.Compact(phi::GPUPlace());
  // The previous compacted tail mapped-free block is a valid source again
  // after its BackingMap meta is refreshed to the destination layout. This
  // call may therefore remap more than only the newly freed second block.
  EXPECT_GE(second_compacted, underlying->handle_size());
  EXPECT_EQ(second_compacted % underlying->handle_size(), 0UL);

  first_block = allocator.all_blocks().begin();
  ASSERT_NE(first_block, allocator.all_blocks().end());
  ASSERT_TRUE(first_block->IsUnmappedFree());
  EXPECT_EQ(first_block->ptr_, first_ptr);
  EXPECT_EQ(first_block->size_, underlying->handle_size() * 2);

  auto next = std::next(first_block);
  ASSERT_NE(next, allocator.all_blocks().end());
  EXPECT_FALSE(next->IsUnmappedFree());
  EXPECT_EQ(reinterpret_cast<uint8_t*>(first_ptr) + underlying->handle_size(),
            second_ptr);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactSkipsPartialFreeHandle) {
  ScopedVLogLevel vlog_guard(4);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(256UL);
  ASSERT_NE(allocation, nullptr);

  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  const size_t remapped = allocator.Compact(phi::GPUPlace());
  EXPECT_EQ(remapped, 0UL);

  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  auto it = allocator.all_blocks().begin();
  ASSERT_EQ(it->type_, BlockType::kActive);
  ++it;
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->ptr_,
            reinterpret_cast<uint8_t*>(allocation->ptr()) + allocation->size());
  EXPECT_EQ(it->size_, underlying->handle_size() - 256UL);
  ExpectBlockView(*it);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactSkipsMappedFreeTail) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);
  MarkRemapSafeForTest(allocation.get());
  allocation.reset();

  const size_t tail_offset = underlying->tail_offset();
  EXPECT_EQ(allocator.Compact(phi::GPUPlace()), 0UL);
  EXPECT_EQ(underlying->tail_offset(), tail_offset);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactRollsBackCommitException) {
  ScopedVLogLevel vlog_guard(4);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);
  const size_t handle_size = underlying->handle_size();

  auto first = allocator.Allocate(handle_size);
  auto middle = allocator.Allocate(handle_size);
  auto separator = allocator.Allocate(handle_size);
  auto last = allocator.Allocate(handle_size);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(separator, nullptr);
  ASSERT_NE(last, nullptr);
  MarkRemapSafeForTest(middle.get());
  auto* middle_ptr = middle->ptr();
  auto* last_ptr = last->ptr();
  middle.reset();
  last.reset();

  auto source_pages = underlying->CollectRemapSourcePages(
      {{reinterpret_cast<VMMDevicePtr>(middle_ptr), handle_size}}, handle_size);
  ASSERT_EQ(source_pages.size(), 1UL);
  const VMMDevicePtr destination_va =
      underlying->virtual_mem_base() + underlying->tail_offset();
  const size_t tail_offset = underlying->tail_offset();

  FreeBlockRemapCompactor compactor(
      underlying, PoolType::kLarge, [](std::vector<DecoratedAllocationPtr>*) {
        throw std::runtime_error("injected compact commit failure");
      });
  EXPECT_THROW(
      compactor.Compact(&allocator.all_blocks_, handle_size, source_pages),
      std::runtime_error);

  EXPECT_EQ(underlying->tail_offset(), tail_offset);
  EXPECT_TRUE(underlying->IsRangeUnmapped(destination_va, handle_size));
  for (const auto& block : allocator.all_blocks()) {
    EXPECT_FALSE(block.ContainsVARange(destination_va, handle_size));
  }
  allocator.RebuildFreeBlockIndex();
  ASSERT_EQ(allocator.free_blocks_.size(), 2UL);
  EXPECT_TRUE(allocator.unmapped_free_blocks_.empty());
  const auto* restored_source = FindBlockByPtr(allocator, middle_ptr);
  ASSERT_NE(restored_source, nullptr);
  EXPECT_TRUE(restored_source->IsMappedFree());
  const auto* original_tail = FindBlockByPtr(allocator, last_ptr);
  ASSERT_NE(original_tail, nullptr);
  EXPECT_TRUE(original_tail->IsMappedFree());
  EXPECT_EQ(original_tail->size_, handle_size);

  auto reused = allocator.Allocate(handle_size);
  ASSERT_NE(reused, nullptr);
  EXPECT_EQ(reused->ptr(), middle_ptr);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     CompactRestoresUnmappedGapAfterCommitException) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);
  const size_t handle_size = underlying->handle_size();

  auto source = allocator.Allocate(handle_size);
  auto gap = allocator.Allocate(handle_size);
  auto anchor = allocator.Allocate(handle_size);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(gap, nullptr);
  ASSERT_NE(anchor, nullptr);
  auto* source_ptr = source->ptr();
  auto* gap_ptr = gap->ptr();

  gap.reset();
  ASSERT_EQ(allocator.Release(phi::GPUPlace()), handle_size);
  MarkRemapSafeForTest(source.get());
  source.reset();

  auto source_pages = underlying->CollectRemapSourcePages(
      {{reinterpret_cast<VMMDevicePtr>(source_ptr), handle_size}}, handle_size);
  ASSERT_EQ(source_pages.size(), 1UL);

  const VMMDevicePtr terminal_va = underlying->virtual_mem_base() +
                                   underlying->virtual_mem_size() - handle_size;
  allocator.all_blocks_.push_back(
      BlockV2::MakeMappedBlock(BlockType::kActive,
                               reinterpret_cast<void*>(terminal_va),
                               handle_size,
                               PoolType::kLarge));

  FreeBlockRemapCompactor compactor(
      underlying, PoolType::kLarge, [](std::vector<DecoratedAllocationPtr>*) {
        throw std::runtime_error("injected compact commit failure");
      });
  EXPECT_THROW(
      compactor.Compact(&allocator.all_blocks_, handle_size, source_pages),
      std::runtime_error);

  EXPECT_TRUE(underlying->IsRangeUnmapped(
      reinterpret_cast<VMMDevicePtr>(gap_ptr), handle_size));
  const auto* restored_gap = FindBlockByPtr(allocator, gap_ptr);
  ASSERT_NE(restored_gap, nullptr);
  EXPECT_TRUE(restored_gap->IsUnmappedFree());
  const auto* restored_source = FindBlockByPtr(allocator, source_ptr);
  ASSERT_NE(restored_source, nullptr);
  EXPECT_TRUE(restored_source->IsMappedFree());

  allocator.RebuildFreeBlockIndex();
  ASSERT_EQ(allocator.free_blocks_.size(), 1UL);
  ASSERT_EQ(allocator.unmapped_free_blocks_.size(), 1UL);
  EXPECT_EQ(allocator.free_blocks_.begin()->second->ptr_, source_ptr);
  EXPECT_EQ(allocator.unmapped_free_blocks_.begin()->second->ptr_, gap_ptr);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, BoundedCompactSkipsLargeFreeBlock) {
  ScopedVLogLevel vlog_guard(4);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t handle_size = underlying->handle_size();
  auto allocation = allocator.Allocate(handle_size);
  ASSERT_NE(allocation, nullptr);
  allocation.reset();

  EXPECT_EQ(allocator.RemapForAllocation(phi::GPUPlace(), handle_size), 0UL);
  ASSERT_EQ(allocator.all_blocks().size(), 1UL);
  EXPECT_TRUE(allocator.all_blocks().front().IsMappedFree());
}

TEST(VMMAutoGrowthBestFitAllocatorV2, BoundedCompactSkipsTailOnlyFree) {
  ScopedVLogLevel vlog_guard(4);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t handle_size = underlying->handle_size();
  auto active = allocator.Allocate(handle_size);
  auto tail = allocator.Allocate(handle_size);
  ASSERT_NE(active, nullptr);
  ASSERT_NE(tail, nullptr);
  tail.reset();

  EXPECT_EQ(allocator.RemapForAllocation(phi::GPUPlace(), handle_size + 1UL),
            0UL);
  ASSERT_EQ(allocator.all_blocks().size(), 2UL);
  EXPECT_TRUE(allocator.all_blocks().back().IsMappedFree());
  EXPECT_EQ(allocator.all_blocks().back().size_, handle_size);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, BoundedCompactSkipsInsufficientSource) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->handle_size());
  auto second = allocator.Allocate(underlying->handle_size());
  auto separator = allocator.Allocate(underlying->handle_size());
  auto third = allocator.Allocate(underlying->handle_size());
  auto fourth = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(separator, nullptr);
  ASSERT_NE(third, nullptr);
  ASSERT_NE(fourth, nullptr);

  second.reset();
  third.reset();
  auto partial = allocator.Allocate(256UL);
  ASSERT_NE(partial, nullptr);
  const size_t block_count_before = allocator.all_blocks().size();
  const size_t requested_size = underlying->handle_size() + 1UL;

  const size_t remapped =
      allocator.RemapForAllocation(phi::GPUPlace(), requested_size);
  EXPECT_EQ(remapped, 0UL);
  EXPECT_EQ(allocator.all_blocks().size(), block_count_before);
  for (const auto& block : allocator.all_blocks()) {
    EXPECT_FALSE(block.IsUnmappedFree());
  }
}

TEST(VMMAutoGrowthBestFitAllocatorV2, BoundedCompactUsesDriverTopUp) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t handle_size = underlying->handle_size();
  auto first = allocator.Allocate(handle_size);
  auto movable = allocator.Allocate(handle_size);
  auto tail_guard = allocator.Allocate(handle_size);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(movable, nullptr);
  ASSERT_NE(tail_guard, nullptr);

  auto* movable_ptr = movable->ptr();
  MarkRemapSafeForTest(movable.get());
  movable.reset();

  const size_t requested_size = 3UL * handle_size;
  const size_t remapped =
      allocator.RemapForAllocation(phi::GPUPlace(), requested_size);
  EXPECT_EQ(remapped, handle_size);

  bool found_old_source = false;
  bool found_tail_free = false;
  for (const auto& block : allocator.all_blocks()) {
    if (block.ptr_ == movable_ptr) {
      found_old_source = true;
      EXPECT_TRUE(block.IsUnmappedFree());
      EXPECT_EQ(block.size_, handle_size);
    }
    if (block.IsMappedFree() && block.ptr_ != movable_ptr) {
      found_tail_free = true;
      EXPECT_EQ(block.size_, handle_size);
    }
  }
  EXPECT_TRUE(found_old_source);
  EXPECT_TRUE(found_tail_free);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, BoundedCompactCountsTailFree) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t handle_size = underlying->handle_size();
  auto unmapped_first = allocator.Allocate(handle_size);
  auto unmapped_second = allocator.Allocate(handle_size);
  auto separator = allocator.Allocate(handle_size);
  auto movable = allocator.Allocate(handle_size);
  auto tail_guard = allocator.Allocate(handle_size);
  ASSERT_NE(unmapped_first, nullptr);
  ASSERT_NE(unmapped_second, nullptr);
  ASSERT_NE(separator, nullptr);
  ASSERT_NE(movable, nullptr);
  ASSERT_NE(tail_guard, nullptr);

  auto* unmapped_ptr = unmapped_first->ptr();
  MarkRemapSafeForTest(unmapped_first.get());
  MarkRemapSafeForTest(unmapped_second.get());
  unmapped_first.reset();
  unmapped_second.reset();
  ASSERT_EQ(allocator.Compact(phi::GPUPlace()), 2UL * handle_size);

  MarkRemapSafeForTest(movable.get());
  movable.reset();
  const size_t requested_size = 2UL * handle_size + 1UL;
  const size_t remapped =
      allocator.RemapForAllocation(phi::GPUPlace(), requested_size);
  // The existing 2-handle tail free range already contributes to this
  // 3-handle aligned request. Only the one-handle non-tail gap must move.
  EXPECT_EQ(remapped, handle_size);

  bool found_unmapped_range = false;
  bool found_movable_source = false;
  for (const auto& block : allocator.all_blocks()) {
    if (block.ptr_ == unmapped_ptr) {
      found_unmapped_range = true;
      EXPECT_TRUE(block.IsUnmappedFree());
      EXPECT_EQ(block.size_, 2UL * handle_size);
    }
    if (block.IsUnmappedFree() && block.size_ == handle_size &&
        block.ptr_ != unmapped_ptr) {
      found_movable_source = true;
    }
  }
  EXPECT_TRUE(found_unmapped_range);
  EXPECT_TRUE(found_movable_source);

  auto recovered = allocator.Allocate(requested_size);
  ASSERT_NE(recovered, nullptr);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, ExplicitCompactAll) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->handle_size());
  auto second = allocator.Allocate(underlying->handle_size());
  auto separator = allocator.Allocate(underlying->handle_size());
  auto third = allocator.Allocate(underlying->handle_size());
  auto fourth = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(separator, nullptr);
  ASSERT_NE(third, nullptr);
  ASSERT_NE(fourth, nullptr);

  auto* third_ptr = third->ptr();
  MarkRemapSafeForTest(second.get());
  MarkRemapSafeForTest(third.get());
  second.reset();
  third.reset();
  auto partial = allocator.Allocate(256UL);
  ASSERT_NE(partial, nullptr);

  const size_t remapped = allocator.Compact(phi::GPUPlace());

  EXPECT_EQ(remapped, underlying->handle_size());
  bool found_third_unmapped = false;
  for (const auto& block : allocator.all_blocks()) {
    if (block.ptr_ == third_ptr) {
      found_third_unmapped = block.IsUnmappedFree();
      EXPECT_EQ(block.size_, underlying->handle_size());
    }
  }
  EXPECT_TRUE(found_third_unmapped);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactSkipsBackingMapPendingEvent) {
  ScopedVLogLevel vlog_guard(4);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->handle_size());
  auto tail_guard = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);
  ASSERT_NE(tail_guard, nullptr);

  gpuStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  BusyWaitKernel<<<1, 1, 0, stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  gpuEvent_t event;
  ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            cudaSuccess);
  ASSERT_EQ(cudaEventRecord(event, stream), cudaSuccess);
  auto guard = std::make_shared<CUDAEventGuard>(event);
  auto* remap_allocation =
      dynamic_cast<VMMRemapEventAllocation*>(allocation.get());
  ASSERT_NE(remap_allocation, nullptr);
  ASSERT_TRUE(remap_allocation->SetVMMRemapEvent(stream, guard));

  allocation.reset();
  EXPECT_EQ(allocator.Compact(phi::GPUPlace()), 0UL);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(allocator.Compact(phi::GPUPlace()), underlying->handle_size());

  ASSERT_EQ(allocator.all_blocks().size(), 3UL);
  auto it = allocator.all_blocks().begin();
  ASSERT_EQ(it->type_, BlockType::kUnmappedFree);
  ++it;
  ASSERT_EQ(it->type_, BlockType::kActive);
  ++it;
  ASSERT_EQ(it->type_, BlockType::kFree);
  EXPECT_EQ(it->size_, underlying->handle_size());
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactSkipsPendingOwningStream) {
  ScopedVLogLevel vlog_guard(4);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto allocation = allocator.Allocate(underlying->handle_size());
  auto tail_guard = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(allocation, nullptr);
  ASSERT_NE(tail_guard, nullptr);

  gpuStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  BusyWaitKernel<<<1, 1, 0, stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  auto* remap_allocation =
      dynamic_cast<VMMRemapEventAllocation*>(allocation.get());
  ASSERT_NE(remap_allocation, nullptr);
  ASSERT_TRUE(remap_allocation->SetVMMRemapEvent(stream, nullptr));

  allocation.reset();
  EXPECT_EQ(allocator.Compact(phi::GPUPlace()), 0UL);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(allocator.Compact(phi::GPUPlace()), underlying->handle_size());

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, OOMRemapSkipsPendingOwningStreamEvents) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->handle_size());
  auto separator = allocator.Allocate(underlying->handle_size());
  auto second = allocator.Allocate(underlying->handle_size());
  auto tail_guard = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(separator, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(tail_guard, nullptr);

  gpuStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  BusyWaitKernel<<<1, 1, 0, stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  auto* first_remap = dynamic_cast<VMMRemapEventAllocation*>(first.get());
  auto* second_remap = dynamic_cast<VMMRemapEventAllocation*>(second.get());
  ASSERT_NE(first_remap, nullptr);
  ASSERT_NE(second_remap, nullptr);
  ASSERT_TRUE(first_remap->SetVMMRemapEvent(stream, nullptr));
  ASSERT_TRUE(second_remap->SetVMMRemapEvent(stream, nullptr));
  first.reset();
  second.reset();

  EXPECT_EQ(allocator.RemapForAllocation(phi::GPUPlace(),
                                         2UL * underlying->handle_size()),
            0UL);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(allocator.RemapForAllocation(phi::GPUPlace(),
                                         2UL * underlying->handle_size()),
            2UL * underlying->handle_size());
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2,
     BoundedRemapSkipsPendingBlockAndUsesLaterReadyBlock) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);
  const size_t handle_size = underlying->handle_size();

  auto pending = allocator.Allocate(handle_size);
  auto separator = allocator.Allocate(handle_size);
  auto ready = allocator.Allocate(handle_size);
  auto tail_active = allocator.Allocate(256UL);
  ASSERT_NE(pending, nullptr);
  ASSERT_NE(separator, nullptr);
  ASSERT_NE(ready, nullptr);
  ASSERT_NE(tail_active, nullptr);

  gpuStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  BusyWaitKernel<<<1, 1, 0, stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  auto* pending_remap = dynamic_cast<VMMRemapEventAllocation*>(pending.get());
  ASSERT_NE(pending_remap, nullptr);
  ASSERT_TRUE(pending_remap->SetVMMRemapEvent(stream, nullptr));
  MarkRemapSafeForTest(ready.get());
  pending.reset();
  ready.reset();

  // The partial tail leaves a 256-byte deficit beyond one handle. Rounded to
  // backing granularity, bounded remap needs one source page. The earlier
  // pending block must not consume that page budget before the later ready
  // block is considered.
  EXPECT_EQ(allocator.RemapForAllocation(phi::GPUPlace(), handle_size + 256UL),
            handle_size);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactUsesBlockListTailPlacement) {
  ScopedVLogLevel vlog_guard(4);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  auto first = allocator.Allocate(underlying->handle_size());
  auto middle = allocator.Allocate(underlying->handle_size());
  auto last = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(last, nullptr);

  MarkRemapSafeForTest(middle.get());
  middle.reset();
  const size_t remaining_tail =
      underlying->virtual_mem_size() - underlying->tail_offset();
  underlying->AdvanceTailOffset(remaining_tail);

  const size_t remapped = allocator.Compact(phi::GPUPlace());
  EXPECT_EQ(remapped, underlying->handle_size());

  void* remapped_free_ptr = nullptr;
  for (const auto& block : allocator.all_blocks()) {
    if (block.type_ == BlockType::kFree &&
        block.size_ == underlying->handle_size()) {
      ExpectBlockView(block);
      remapped_free_ptr = block.ptr_;
      break;
    }
  }
  ASSERT_NE(remapped_free_ptr, nullptr)
      << "expected one remapped tail free block";

  auto remap_sources = underlying->CollectRemapSourcePages(
      {{reinterpret_cast<VMMDevicePtr>(remapped_free_ptr),
        underlying->handle_size()}},
      underlying->handle_size());
  ASSERT_EQ(remap_sources.size(), 1UL);
  EXPECT_EQ(remap_sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kRemapDestinationOwned);

  auto remapped_active = allocator.Allocate(underlying->handle_size());
  ASSERT_NE(remapped_active, nullptr);
  std::vector<BlockPart> ipc_parts;
  EXPECT_TRUE(allocator.CollectTensorParts(
      remapped_active->ptr(), underlying->handle_size(), &ipc_parts));
  ASSERT_EQ(ipc_parts.size(), 1UL);
  EXPECT_EQ(ipc_parts[0].chunk->base,
            reinterpret_cast<VMMDevicePtr>(remapped_active->ptr()));
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactKeepsMappedFreeBlocksAsViews) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t handle_size = underlying->handle_size();
  auto large = allocator.Allocate(3UL * handle_size);
  auto tail_guard = allocator.Allocate(handle_size);
  ASSERT_NE(large, nullptr);
  ASSERT_NE(tail_guard, nullptr);
  MarkRemapSafeForTest(large.get());
  large.reset();

  auto prefix = allocator.Allocate(256);
  ASSERT_NE(prefix, nullptr);

  const size_t remapped = allocator.Compact(phi::GPUPlace());
  ASSERT_EQ(remapped, 2UL * handle_size);

  for (const auto& block : allocator.all_blocks()) {
    if (block.IsMappedFree()) {
      ExpectBlockView(block);
    }
  }

  prefix.reset();
  tail_guard.reset();
  for (const auto& block : allocator.all_blocks()) {
    if (block.IsMappedFree()) {
      ExpectBlockView(block);
    }
  }

  const size_t tail_before_release = underlying->tail_offset();
  EXPECT_GT(allocator.Release(phi::GPUPlace()), 0UL);
  std::vector<std::pair<VMMDevicePtr, size_t>> released_range = {
      {underlying->virtual_mem_base(), tail_before_release}};
  EXPECT_TRUE(underlying->CollectMappedPages(released_range, 0).empty());
}

TEST(VMMAutoGrowthBestFitAllocatorV2, CompactUsesUnmappedTargets) {
  ScopedVLogLevel vlog_guard(4);
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t handle_size = underlying->handle_size();
  auto target_a = allocator.Allocate(handle_size);
  auto source_a = allocator.Allocate(handle_size);
  auto target_b = allocator.Allocate(handle_size);
  auto source_b = allocator.Allocate(handle_size);
  ASSERT_NE(target_a, nullptr);
  ASSERT_NE(source_a, nullptr);
  ASSERT_NE(target_b, nullptr);
  ASSERT_NE(source_b, nullptr);
  auto* target_a_ptr = target_a->ptr();
  auto* source_a_ptr = source_a->ptr();
  auto* target_b_ptr = target_b->ptr();
  auto* source_b_ptr = source_b->ptr();

  MarkRemapSafeForTest(target_a.get());
  MarkRemapSafeForTest(target_b.get());
  target_a.reset();
  target_b.reset();
  ASSERT_EQ(allocator.Compact(phi::GPUPlace()), 2UL * handle_size);
  auto tail_active = allocator.Allocate(2UL * handle_size);
  ASSERT_NE(tail_active, nullptr);

  auto hidden_tail_mapping = underlying->AppendWithBlock(handle_size);
  ASSERT_TRUE(hidden_tail_mapping.HasAllocation());

  MarkRemapSafeForTest(source_a.get());
  MarkRemapSafeForTest(source_b.get());
  source_a.reset();
  source_b.reset();
  EXPECT_EQ(allocator.RemapForAllocation(phi::GPUPlace(), 2UL * handle_size),
            2UL * handle_size);

  const auto* target_a_block = FindBlockByPtr(allocator, target_a_ptr);
  ASSERT_NE(target_a_block, nullptr);
  EXPECT_TRUE(target_a_block->IsFree());
  ExpectBlockView(*target_a_block);

  const auto* target_b_block = FindBlockByPtr(allocator, target_b_ptr);
  ASSERT_NE(target_b_block, nullptr);
  EXPECT_TRUE(target_b_block->IsFree());
  ExpectBlockView(*target_b_block);

  const auto* source_a_block = FindBlockByPtr(allocator, source_a_ptr);
  ASSERT_NE(source_a_block, nullptr);
  EXPECT_TRUE(source_a_block->IsUnmappedFree());
  EXPECT_EQ(source_a_block->size(), handle_size);

  const auto* source_b_block = FindBlockByPtr(allocator, source_b_ptr);
  ASSERT_NE(source_b_block, nullptr);
  EXPECT_TRUE(source_b_block->IsUnmappedFree());
  EXPECT_EQ(source_b_block->size(), handle_size);
}

TEST(VMMAutoGrowthBestFitAllocatorV2, RemapUsesStaleUnmappedRange) {
  auto underlying = CreateUnderlyingAllocator();
  VMMAutoGrowthBestFitAllocatorV2 allocator(
      underlying, 256, phi::GPUPlace(), PoolType::kLarge);

  const size_t handle_size = underlying->handle_size();
  auto large = allocator.Allocate(4UL * handle_size);
  ASSERT_NE(large, nullptr);
  auto* base_ptr = large->ptr();
  large.reset();

  auto target = allocator.Allocate(handle_size);
  auto first_source = allocator.Allocate(handle_size);
  auto second_source = allocator.Allocate(handle_size);
  auto tail_guard_source = allocator.Allocate(handle_size);
  ASSERT_NE(target, nullptr);
  ASSERT_NE(first_source, nullptr);
  ASSERT_NE(second_source, nullptr);
  ASSERT_NE(tail_guard_source, nullptr);
  EXPECT_EQ(target->ptr(), base_ptr);
  auto* first_source_ptr = first_source->ptr();
  auto* second_source_ptr = second_source->ptr();

  MarkRemapSafeForTest(first_source.get());
  first_source.reset();
  ASSERT_EQ(allocator.RemapForAllocation(phi::GPUPlace(), handle_size + 1UL),
            handle_size);

  auto consume_tail_destination = allocator.Allocate(handle_size);
  ASSERT_NE(consume_tail_destination, nullptr);
  auto hidden_tail_mapping = underlying->AppendWithBlock(handle_size);
  ASSERT_TRUE(hidden_tail_mapping.HasAllocation());

  MarkRemapSafeForTest(second_source.get());
  second_source.reset();
  ASSERT_EQ(allocator.RemapForAllocation(phi::GPUPlace(), handle_size + 1UL),
            handle_size);

  const auto* first_source_block = FindBlockByPtr(allocator, first_source_ptr);
  ASSERT_NE(first_source_block, nullptr);
  EXPECT_TRUE(first_source_block->IsFree());

  const auto* second_source_block =
      FindBlockByPtr(allocator, second_source_ptr);
  ASSERT_NE(second_source_block, nullptr);
  EXPECT_TRUE(second_source_block->IsUnmappedFree());

  target.reset();
  tail_guard_source.reset();
  consume_tail_destination.reset();
  hidden_tail_mapping.allocation.reset();
  EXPECT_GT(allocator.Release(phi::GPUPlace()), 0UL);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
