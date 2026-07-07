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

#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.h"

#if defined(PADDLE_WITH_CUDA)

#include <exception>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"

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

void VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::Add(
    DecoratedAllocationPtr allocation) {
  allocations_.emplace_back(std::move(allocation));
  auto it = std::prev(allocations_.end());
  auto* begin = Begin(*it);
  PADDLE_ENFORCE_EQ(
      allocations_by_ptr_.emplace(begin, it).second,
      true,
      common::errors::AlreadyExists(
          "Duplicate underlying allocation base %p in VMM V2 registry.",
          begin));
}

namespace {

bool RangesOverlap(void* lhs_ptr,
                   size_t lhs_size,
                   void* rhs_ptr,
                   size_t rhs_size) {
  const auto* lhs_begin = reinterpret_cast<const uint8_t*>(lhs_ptr);
  const auto* lhs_end = lhs_begin + lhs_size;
  const auto* rhs_begin = reinterpret_cast<const uint8_t*>(rhs_ptr);
  const auto* rhs_end = rhs_begin + rhs_size;
  return lhs_end > rhs_begin && rhs_end > lhs_begin;
}

}  // namespace

uint8_t* VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::Begin(
    const DecoratedAllocationPtr& allocation) {
  return reinterpret_cast<uint8_t*>(allocation->ptr());
}

uint8_t* VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::End(
    const DecoratedAllocationPtr& allocation) {
  return Begin(allocation) + allocation->size();
}

bool VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::HasOverlap(
    void* ptr, size_t size) const {
  auto* begin = reinterpret_cast<uint8_t*>(ptr);
  auto* end = begin + size;
  auto it = allocations_by_ptr_.lower_bound(begin);
  if (it != allocations_by_ptr_.begin()) {
    auto prev = std::prev(it);
    if (End(*prev->second) > begin) {
      return true;
    }
  }
  return it != allocations_by_ptr_.end() && it->first < end;
}

bool VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::Overlaps(
    void* ptr, size_t size) const {
  return HasOverlap(ptr, size);
}

VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::iterator
VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::Erase(
    iterator it) {
  allocations_by_ptr_.erase(Begin(*it));
  return allocations_.erase(it);
}

VMMAutoGrowthBestFitAllocatorV2::VMMAutoGrowthBestFitAllocatorV2(
    const std::shared_ptr<CUDAVirtualMemAllocatorV2>& underlying_allocator,
    size_t alignment,
    const GPUPlace& place,
    PoolType pool_type)
    : underlying_allocator_(underlying_allocator),
      alignment_(alignment),
      place_(place),
      pool_type_(pool_type) {}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocateImpl(size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  const size_t requested_size = AlignedSize(size, alignment_);
  if (auto* allocation = AllocFromFreeBlocks(requested_size)) {
    return allocation;
  }
  if (auto* allocation = AllocFromUnmappedFreeBlocks(requested_size)) {
    return allocation;
  }

  // Tail reuse: if the last block in the address space is FREE, detach it
  // and only request the difference from the underlying allocator. The
  // underlying VMM provider maps new handles at a monotonically increasing
  // VA cursor, so the new allocation is guaranteed to be contiguous with
  // the tail FREE block.
  bool has_tail_reuse = false;
  size_t tail_reuse_size = 0;
  BlockV2 combined_free_block;
  if (!all_blocks_.empty()) {
    auto tail_it = std::prev(all_blocks_.end());
    if (CanIndexFreeBlock(*tail_it)) {
      has_tail_reuse = true;
      tail_reuse_size = tail_it->size_;
      EraseFreeBlock(tail_it);
      combined_free_block = std::move(*tail_it);
      all_blocks_.erase(tail_it);
    }
  }

  const size_t grow_size = (requested_size > tail_reuse_size)
                               ? (requested_size - tail_reuse_size)
                               : 0;
  auto restore_tail_free_block = [&] {
    if (has_tail_reuse) {
      auto restored_it =
          all_blocks_.insert(all_blocks_.end(), std::move(combined_free_block));
      InsertFreeBlock(restored_it);
    }
  };

  // Grow: obtain a new raw allocation from the bottom VMM provider.
  // If cuMemCreate fails due to physical memory exhaustion (CU error 2),
  // the driver-level allocator throws EnforceNotMet. Convert it to BadAlloc so
  // the outer retry path can handle it.
  CUDAVirtualMemAllocatorV2::AllocationWithBlock grow_alloc;
  if (grow_size > 0) {
    try {
      grow_alloc = underlying_allocator_->AppendWithBlock(grow_size);
    } catch (const BadAlloc& bad_alloc) {
      restore_tail_free_block();
      PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
          "VMM V2 best-fit allocator (pool %d) failed to grow by %zu bytes.\n"
          "Underlying VMM allocation failure:\n%s",
          static_cast<int>(pool_type_),
          grow_size,
          bad_alloc.what()));
    } catch (const std::exception& e) {
      restore_tail_free_block();
      PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
          "VMM V2 best-fit allocator (pool %d) failed to grow by %zu bytes.\n"
          "Underlying VMM allocation exception:\n%s",
          static_cast<int>(pool_type_),
          grow_size,
          e.what()));
    } catch (...) {
      restore_tail_free_block();
      PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
          "VMM V2 best-fit allocator (pool %d) failed to grow by %zu bytes "
          "with an unknown underlying VMM allocation exception.",
          static_cast<int>(pool_type_),
          grow_size));
    }
  }

  size_t total_new_size = tail_reuse_size;

  if (grow_alloc.HasAllocation()) {
    BlockV2 grow_block = AdoptBackingBlock(&grow_alloc);
    total_new_size += grow_block.size_;
    if (has_tail_reuse) {
      combined_free_block.MergeAdjacentBlock(grow_block);
    } else {
      combined_free_block = std::move(grow_block);
    }
  }

  const size_t remaining_size = total_new_size - requested_size;

  BlockV2 block =
      combined_free_block.MakeMappedActiveSubBlock(0, requested_size);
  auto it = all_blocks_.insert(all_blocks_.end(), std::move(block));

  if (remaining_size > 0) {
    BlockV2 remaining_block = combined_free_block.MakeMappedFreeSubBlock(
        requested_size, remaining_size);
    auto remain_it =
        all_blocks_.insert(std::next(it), std::move(remaining_block));
    InsertFreeBlock(remain_it);
  }

  return new VMMAutoGrowthBestFitBlockAllocationV2(it, place_, this);  // NOLINT
}

void VMMAutoGrowthBestFitAllocatorV2::FreeImpl(phi::Allocation* allocation) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto* wrapped_allocation =
      static_cast<VMMAutoGrowthBestFitBlockAllocationV2*>(allocation);
  auto it = wrapped_allocation->block_it();
  PADDLE_ENFORCE_NE(
      it,
      all_blocks_.end(),
      common::errors::NotFound("Can not find active block for allocation %p in "
                               "VMMAutoGrowthBestFitAllocatorV2.",
                               allocation->ptr()));
  it->MarkFree();
  TryMerge(it);
  delete allocation;
}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocFromFreeBlocks(
    size_t size) {
  auto it = free_blocks_.lower_bound({size, nullptr});
  if (it == free_blocks_.end()) {
    return nullptr;
  }

  auto block_it = it->second;
  PADDLE_ENFORCE_EQ(
      CanIndexFreeBlock(*block_it),
      true,
      common::errors::PreconditionNotMet(
          "VMM V2 free block index points to a non-reusable block."));
  free_blocks_.erase(it);

  if (block_it->size_ > size) {
    const size_t remaining_size = block_it->size_ - size;
    BlockV2 remaining_block =
        block_it->MakeMappedFreeSubBlock(size, remaining_size);
    block_it->TrimToPrefix(size);
    auto remain_it =
        all_blocks_.insert(std::next(block_it), std::move(remaining_block));
    InsertFreeBlock(remain_it);
  }

  block_it->MarkActive();
  return new  // NOLINT
      VMMAutoGrowthBestFitBlockAllocationV2(block_it, place_, this);
}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocFromUnmappedFreeBlocks(
    size_t size) {
  const size_t backing_size =
      AlignedSize(size, underlying_allocator_->handle_size());
  BlockListIt best = all_blocks_.end();
  for (auto iter = unmapped_free_blocks_.lower_bound({backing_size, nullptr});
       iter != unmapped_free_blocks_.end();) {
    auto it = iter->second;
    if (!it->IsUnmappedFree()) {
      iter = unmapped_free_blocks_.erase(iter);
      continue;
    }
    if (RangeOverlapsUnderlying(it->ptr_, backing_size)) {
      VLOG(6) << "VMM V2 AllocFromUnmappedFreeBlocks skip ownership-overlapped "
                 "unmapped-free ptr="
              << it->ptr_ << " backing_size=" << backing_size
              << " block_size=" << it->size_;
      ++iter;
      continue;
    }
    best = it;
    break;
  }
  if (best == all_blocks_.end()) {
    return nullptr;
  }

  const auto unmapped_free_ptr = best->begin_va();
  VLOG(6) << "VMM V2 AllocFromUnmappedFreeBlocks ptr="
          << reinterpret_cast<void*>(unmapped_free_ptr) << " requested=" << size
          << " backing_size=" << backing_size
          << " original_unmapped_free_size=" << best->size_
          << " tail_offset=" << underlying_allocator_->tail_offset();
  CUDAVirtualMemAllocatorV2::AllocationWithBlock unmapped_free_alloc;
  try {
    unmapped_free_alloc = underlying_allocator_->PlaceAtVAWithBlock(
        unmapped_free_ptr, backing_size);
  } catch (const BadAlloc&) {
    // Do not mutate the allocation view if backing cannot be created in this
    // unmapped-free range due to physical memory pressure. The normal grow
    // path will surface the allocation failure if needed. Other exceptions
    // indicate allocator state bugs and must not be hidden as a cache miss.
    return nullptr;
  }

  BlockV2 mapped_block = AdoptBackingBlock(&unmapped_free_alloc);
  PADDLE_ENFORCE_EQ(
      mapped_block.size_,
      backing_size,
      common::errors::InvalidArgument(
          "Unexpected unmapped-free backing size: got %zu, expected %zu.",
          mapped_block.size_,
          backing_size));

  const size_t original_unmapped_free_size = best->size_;
  const PoolType original_pool_type = best->pool_type_;

  EraseUnmappedFreeBlock(best);
  *best = mapped_block.MakeMappedActiveSubBlock(0, size);

  auto insert_pos = std::next(best);
  if (backing_size > size) {
    BlockV2 mapped_remain =
        mapped_block.MakeMappedFreeSubBlock(size, backing_size - size);
    auto free_it = all_blocks_.insert(insert_pos, std::move(mapped_remain));
    InsertFreeBlock(free_it);
    insert_pos = std::next(free_it);
  }

  if (original_unmapped_free_size > backing_size) {
    BlockV2 tail_unmapped_free = BlockV2::MakeUnmappedFreeBlock(
        reinterpret_cast<uint8_t*>(best->ptr_) + backing_size,
        original_unmapped_free_size - backing_size,
        original_pool_type);
    auto tail_it =
        all_blocks_.insert(insert_pos, std::move(tail_unmapped_free));
    InsertUnmappedFreeBlock(tail_it);
  }

  return new  // NOLINT
      VMMAutoGrowthBestFitBlockAllocationV2(best, place_, this);
}

void VMMAutoGrowthBestFitAllocatorV2::TrackUnderlyingAllocation(
    DecoratedAllocationPtr allocation) {
  underlying_allocations_.Add(std::move(allocation));
}

BlockV2 VMMAutoGrowthBestFitAllocatorV2::AdoptBackingBlock(
    CUDAVirtualMemAllocatorV2::AllocationWithBlock* allocation_with_block) {
  PADDLE_ENFORCE_NOT_NULL(
      allocation_with_block,
      common::errors::InvalidArgument(
          "AllocationWithBlock must not be null when adopting block."));
  BlockV2 block = allocation_with_block->TakeBlock();
  auto allocation = static_unique_ptr_cast<Allocation>(
      allocation_with_block->TakeAllocation());
  TrackUnderlyingAllocation(std::move(allocation));
  return block;
}

bool VMMAutoGrowthBestFitAllocatorV2::RangeOverlapsUnderlying(
    void* ptr, size_t size) const {
  return underlying_allocations_.Overlaps(ptr, size);
}

bool VMMAutoGrowthBestFitAllocatorV2::HasReleasableIdleUnderlying() const {
  for (const auto& allocation : underlying_allocations_) {
    auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
    if (CanReleaseIdleUnderlying(base, allocation->size())) {
      return true;
    }
  }
  return false;
}

bool VMMAutoGrowthBestFitAllocatorV2::CanReleaseIdleUnderlying(
    uint8_t* base, size_t size) const {
  if (!IsRangeEntirelyFree(base, size)) {
    return false;
  }
  return underlying_allocator_->IsRangeReleasable(
      reinterpret_cast<VMMDevicePtr>(base), size);
}

bool VMMAutoGrowthBestFitAllocatorV2::TryReleaseIdleUnderlying(
    UnderlyingAllocationRegistry::iterator* alloc_it, uint64_t* released) {
  auto* allocation = (**alloc_it).get();
  auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
  const size_t alloc_size = allocation->size();
  if (!CanReleaseIdleUnderlying(base, alloc_size)) {
    return false;
  }

  ReplaceRangeWithUnmappedFree(base, alloc_size);
  *released += alloc_size;
  VLOG(5) << "VMM V2 pool " << static_cast<int>(pool_type_)
          << " released idle chunk: " << alloc_size << " bytes";
  *alloc_it = underlying_allocations_.Erase(*alloc_it);
  return true;
}

bool VMMAutoGrowthBestFitAllocatorV2::CanIndexFreeBlock(
    const BlockV2& block) const {
  return block.IsMappedFree();
}

void VMMAutoGrowthBestFitAllocatorV2::InsertFreeBlock(BlockListIt it) {
  if (!CanIndexFreeBlock(*it)) {
    return;
  }
  EmplaceOrEnforce(
      &free_blocks_, std::make_pair(it->size_, it->ptr_), it, "free_blocks_");
}

void VMMAutoGrowthBestFitAllocatorV2::EraseFreeBlock(BlockListIt it) {
  free_blocks_.erase({it->size_, it->ptr_});
}

void VMMAutoGrowthBestFitAllocatorV2::InsertUnmappedFreeBlock(BlockListIt it) {
  if (!it->IsUnmappedFree()) {
    return;
  }
  EmplaceOrEnforce(&unmapped_free_blocks_,
                   std::make_pair(it->size_, it->ptr_),
                   it,
                   "unmapped_free_blocks_");
}

void VMMAutoGrowthBestFitAllocatorV2::EraseUnmappedFreeBlock(BlockListIt it) {
  unmapped_free_blocks_.erase({it->size_, it->ptr_});
}

void VMMAutoGrowthBestFitAllocatorV2::TryMerge(BlockListIt it) {
  // Only adjacent FREE blocks are merged here. ACTIVE blocks are never touched,
  // and unmapped-free blocks remain as explicit holes for later reuse.
  // all_blocks_ is the full VA-ordered block list, so adjacency is checked
  // against neighboring entries in that list.
  if (it != all_blocks_.begin()) {
    auto prev = std::prev(it);
    if (prev->CanMergeAdjacentFreeBlock(*it)) {
      EraseFreeBlock(prev);
      prev->MergeAdjacentBlock(*it);
      all_blocks_.erase(it);
      it = prev;
    }
  }

  auto next = std::next(it);
  if (next != all_blocks_.end() && it->CanMergeAdjacentFreeBlock(*next)) {
    EraseFreeBlock(next);
    it->MergeAdjacentBlock(*next);
    all_blocks_.erase(next);
  }

  InsertFreeBlock(it);
}

void VMMAutoGrowthBestFitAllocatorV2::TryMergeUnmappedFree(BlockListIt it) {
  if (it == all_blocks_.end() || !it->IsUnmappedFree()) {
    return;
  }

  if (it != all_blocks_.begin()) {
    auto prev = std::prev(it);
    if (prev->CanMergeAdjacentUnmappedFreeBlock(*it)) {
      EraseUnmappedFreeBlock(prev);
      EraseUnmappedFreeBlock(it);
      prev->MergeAdjacentUnmappedFreeBlock(*it);
      all_blocks_.erase(it);
      it = prev;
      InsertUnmappedFreeBlock(it);
    }
  }

  auto next = std::next(it);
  if (next != all_blocks_.end() &&
      it->CanMergeAdjacentUnmappedFreeBlock(*next)) {
    EraseUnmappedFreeBlock(it);
    EraseUnmappedFreeBlock(next);
    it->MergeAdjacentUnmappedFreeBlock(*next);
    all_blocks_.erase(next);
    InsertUnmappedFreeBlock(it);
  }
}

// ---------------------------------------------------------------------------
// ReleaseImpl / FreeIdleChunks: release underlying allocations whose entire
// VA range is covered by FREE blocks back to the CUDA VMM driver.
//
// Because TryMerge may have merged FREE blocks across allocation boundaries,
// we must split the spanning block at the allocation edges, release the
// backing, and keep the released VA range as explicit unmapped-free space for
// later reuse.
// ---------------------------------------------------------------------------

uint64_t VMMAutoGrowthBestFitAllocatorV2::ReleaseImpl(
    const Place& place UNUSED) {
  std::lock_guard<SpinLock> guard(spinlock_);
  if (!HasReleasableIdleUnderlying()) {
    return 0;
  }
  // FreeIdleChunks may release CUDA VMM mappings and physical handles. Those
  // driver calls are not ordered by the stream-safe wrapper, so wait before
  // making any previously returned VA range invalid.
  platform::CUDADeviceGuard device_guard(place_.device);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  return FreeIdleChunks();
}

uint64_t VMMAutoGrowthBestFitAllocatorV2::FreeIdleChunks() {
  uint64_t released = 0;

  for (auto alloc_it = underlying_allocations_.begin();
       alloc_it != underlying_allocations_.end();) {
    if (!TryReleaseIdleUnderlying(&alloc_it, &released)) {
      ++alloc_it;
    }
  }

  TrimTrailingUnmappedFreeBlocks();
  underlying_allocator_->SetTailOffset(ComputeTailOffset());
  return released;
}

void VMMAutoGrowthBestFitAllocatorV2::TrimTrailingUnmappedFreeBlocks() {
  while (!all_blocks_.empty()) {
    auto tail_it = std::prev(all_blocks_.end());
    if (!tail_it->IsUnmappedFree() ||
        underlying_allocations_.Overlaps(tail_it->ptr_, tail_it->size_)) {
      break;
    }
    EraseUnmappedFreeBlock(tail_it);
    all_blocks_.erase(tail_it);
  }
}

size_t VMMAutoGrowthBestFitAllocatorV2::ComputeTailOffset() const {
  for (auto it = all_blocks_.rbegin(); it != all_blocks_.rend(); ++it) {
    if (it->IsUnmappedFree() &&
        !underlying_allocations_.Overlaps(it->ptr_, it->size_)) {
      continue;
    }
    return static_cast<size_t>(it->end_va() -
                               underlying_allocator_->virtual_mem_base());
  }
  return 0;
}

bool VMMAutoGrowthBestFitAllocatorV2::IsRangeEntirelyFree(uint8_t* base,
                                                          size_t size) const {
  auto* end = base + size;
  for (const auto& block : all_blocks_) {
    auto* bptr = block.begin_ptr();
    auto* bend = block.end_ptr();
    if (bend <= base) continue;
    if (bptr >= end) break;
    if (block.IsActive()) {
      return false;
    }
  }
  // Return true when the range contains only FREE/unmapped-free blocks, or
  // when blocks have already been removed by a prior FreeIdleChunks pass.
  return true;
}

void VMMAutoGrowthBestFitAllocatorV2::ReplaceRangeWithUnmappedFree(
    uint8_t* base, size_t size) {
  auto* end = base + size;
  auto erase_free_index = [this](BlockList::iterator it) {
    if (it->IsUnmappedFree()) {
      EraseUnmappedFreeBlock(it);
    } else {
      EraseFreeBlock(it);
    }
  };
  auto insert_free_index = [this](BlockList::iterator it) {
    if (it->IsUnmappedFree()) {
      InsertUnmappedFreeBlock(it);
    } else {
      InsertFreeBlock(it);
    }
  };

  for (auto it = all_blocks_.begin(); it != all_blocks_.end();) {
    auto* bptr = it->begin_ptr();
    auto* bend = it->end_ptr();

    if (bend <= base) {
      ++it;
      continue;
    }
    if (bptr >= end) break;

    // Case 1: block entirely within [base, end): remove it.
    if (bptr >= base && bend <= end) {
      erase_free_index(it);
      it = all_blocks_.erase(it);
      continue;
    }

    // Case 2: block straddles left boundary only: keep left remnant.
    if (bptr < base && bend <= end) {
      const size_t keep = static_cast<size_t>(base - bptr);
      erase_free_index(it);
      it->TrimToPrefix(keep);
      insert_free_index(it);
      ++it;
      continue;
    }

    // Case 3: block straddles right boundary only: keep right remnant.
    if (bptr >= base && bend > end) {
      const size_t trim = static_cast<size_t>(end - bptr);
      const size_t keep = it->size_ - trim;
      erase_free_index(it);
      it->TrimToSuffix(trim, keep);
      insert_free_index(it);
      break;  // nothing more in range
    }

    // Case 4: block fully encompasses [base, end): split into two.
    if (bptr < base && bend > end) {
      const size_t left_size = static_cast<size_t>(base - bptr);
      const size_t right_offset = static_cast<size_t>(end - bptr);
      const size_t right_size = it->size_ - right_offset;
      BlockV2 right =
          it->IsUnmappedFree()
              ? it->MakeUnmappedFreeSubBlock(right_offset, right_size)
              : it->MakeMappedFreeSubBlock(right_offset, right_size);

      erase_free_index(it);
      it->TrimToPrefix(left_size);
      insert_free_index(it);
      auto right_it = all_blocks_.insert(std::next(it), std::move(right));
      insert_free_index(right_it);
      break;  // done
    }

    ++it;
  }

  auto insert_pos = all_blocks_.begin();
  while (insert_pos != all_blocks_.end() && insert_pos->begin_ptr() < base) {
    ++insert_pos;
  }
  auto unmapped_it = all_blocks_.insert(
      insert_pos, BlockV2::MakeUnmappedFreeBlock(base, size, pool_type_));
  InsertUnmappedFreeBlock(unmapped_it);
  TryMergeUnmappedFree(unmapped_it);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
