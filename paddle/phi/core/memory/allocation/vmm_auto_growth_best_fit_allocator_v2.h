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

#pragma once

#include <functional>
#include <list>
#include <map>
#include <memory>

#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"
#include "paddle/phi/core/memory/allocation/vmm_ipc_allocation.h"
#include "paddle/phi/core/memory/mem_visitor.h"

#if defined(PADDLE_WITH_CUDA)

namespace paddle {
namespace memory {
namespace allocation {

using BlockList = std::list<BlockV2>;
using BlockListIt = BlockList::iterator;

class VMMAutoGrowthBestFitAllocatorV2;

class VMMAutoGrowthBestFitBlockAllocationV2 : public Allocation,
                                              public VMMRemapEventAllocation {
 public:
  VMMAutoGrowthBestFitBlockAllocationV2(BlockListIt block_it,
                                        const Place& place,
                                        VMMAutoGrowthBestFitAllocatorV2* owner)
      : Allocation(block_it->ptr_, block_it->ptr_, block_it->size_, place),
        block_it_(block_it),
        owner_(owner) {}

  BlockListIt block_it() const { return block_it_; }
  bool SetVMMRemapEvent(gpuStream_t stream,
                        std::shared_ptr<CUDAEventGuard> event) override;
  gpuStream_t remap_stream() const { return remap_stream_; }
  std::shared_ptr<CUDAEventGuard> TakeRemapEvent() {
    return std::move(remap_event_);
  }

 private:
  BlockListIt block_it_;
  VMMAutoGrowthBestFitAllocatorV2* owner_;
  gpuStream_t remap_stream_{nullptr};
  std::shared_ptr<CUDAEventGuard> remap_event_;
};

class VMMAutoGrowthBestFitAllocatorV2 : public Allocator {
 public:
  VMMAutoGrowthBestFitAllocatorV2(
      const std::shared_ptr<CUDAVirtualMemAllocatorV2>& underlying_allocator,
      size_t alignment,
      const GPUPlace& place,
      PoolType pool_type);

  bool IsAllocThreadSafe() const override { return true; }
  void Accept(AllocatorVisitor* visitor) override { visitor->Visit(this); }

  const BlockList& all_blocks() const { return all_blocks_; }
  BlockList SnapshotAllBlocks() const;
  PoolType pool_type() const { return pool_type_; }
  size_t alignment() const { return alignment_; }

  // Query aggregate free-block statistics for OOM dispatch decisions.
  // total_free = sum of all FREE block sizes, max_free = largest FREE block.
  void GetFreeBlockStats(size_t* total_free, size_t* max_free);

  bool CollectTensorParts(void* ptr,
                          size_t size,
                          std::vector<BlockPart>* parts,
                          bool mark_ipc_exported = true);

  bool SetBlockRemapEvent(void* ptr,
                          gpuStream_t stream,
                          std::shared_ptr<CUDAEventGuard> event);
  bool SetBlockRemapEvent(BlockListIt block_it,
                          gpuStream_t stream,
                          std::shared_ptr<CUDAEventGuard> event);

  // Compacts mapped-free VMM backing for a failed allocation request. A zero
  // request performs explicit unbounded maintenance compaction.
  size_t RemapForAllocation(const Place& place, size_t requested_size);

 protected:
  phi::Allocation* AllocateImpl(size_t size) override;
  size_t CompactImpl(const Place& place) override;
  void FreeImpl(phi::Allocation* allocation) override;
  uint64_t ReleaseImpl(const Place& place) override;

 private:
  struct UnderlyingAllocationRegistry {
    using List = std::list<DecoratedAllocationPtr>;
    using iterator = List::iterator;
    using OverlapPredicate = std::function<bool(const DecoratedAllocationPtr&)>;

    void Add(DecoratedAllocationPtr allocation);
    bool Overlaps(void* ptr, size_t size) const;
    bool AllOverlapsSatisfy(void* ptr,
                            size_t size,
                            const OverlapPredicate& predicate) const;
    bool EraseOverlapsIf(void* ptr,
                         size_t size,
                         const OverlapPredicate& predicate);
    iterator begin() { return allocations_.begin(); }
    iterator end() { return allocations_.end(); }
    List::const_iterator begin() const { return allocations_.begin(); }
    List::const_iterator end() const { return allocations_.end(); }
    iterator Erase(iterator it);

   private:
    using Index = std::map<uint8_t*, iterator>;
    static uint8_t* Begin(const DecoratedAllocationPtr& allocation);
    static uint8_t* End(const DecoratedAllocationPtr& allocation);
    bool HasOverlap(void* ptr, size_t size) const;

    List allocations_;
    Index allocations_by_ptr_;
  };

  phi::Allocation* AllocFromFreeBlocks(size_t size);
  phi::Allocation* AllocFromUnmappedFreeBlocks(size_t size);
  BlockV2 AdoptBackingBlock(
      CUDAVirtualMemAllocatorV2::AllocationWithBlock* allocation_with_block);
  void TrackUnderlyingAllocation(DecoratedAllocationPtr allocation);
  bool AllocationOwnedByRemapDestination(
      const DecoratedAllocationPtr& allocation,
      void* target_ptr,
      size_t target_size) const;
  bool CanPrepareRemapDestinationRange(void* ptr, size_t size) const;
  bool PrepareRemapDestinationRange(void* ptr, size_t size);
  bool CanReleaseIdleUnderlying(uint8_t* base, size_t size) const;
  bool HasReleasableIdleUnderlying() const;
  bool TryReleaseIdleUnderlying(
      UnderlyingAllocationRegistry::iterator* alloc_it, uint64_t* released);
  bool CanIndexFreeBlock(const BlockV2& block) const;
  void InsertFreeBlock(BlockListIt it);
  void EraseFreeBlock(BlockListIt it);
  void InsertUnmappedFreeBlock(BlockListIt it);
  void EraseUnmappedFreeBlock(BlockListIt it);
  void RebuildFreeBlockIndex();
  void TryMerge(BlockListIt it);
  void TryMergeUnmappedFree(BlockListIt it);
  uint64_t FreeIdleChunks();
  void TrimTrailingUnmappedFreeBlocks();
  size_t ComputeTailOffset() const;
  bool IsRangeEntirelyFree(uint8_t* base, size_t size) const;
  void ReplaceRangeWithUnmappedFree(uint8_t* base, size_t size);

  // Best-fit V2 only grows from the fixed-handle CUDA VMM provider. The
  // bottom allocator returns mapped-free BlockV2 views, while best-fit owns
  // allocation/free-list policy over those block views.
  std::shared_ptr<CUDAVirtualMemAllocatorV2> underlying_allocator_;
  size_t alignment_;
  GPUPlace place_;
  PoolType pool_type_;
  UnderlyingAllocationRegistry underlying_allocations_;
  // Full block list ordered by VA address. This is the source of truth and
  // contains ACTIVE/FREE/UNMAPPED-FREE blocks together.
  BlockList all_blocks_;
  std::map<std::pair<size_t, void*>, BlockListIt> free_blocks_;
  std::map<std::pair<size_t, void*>, BlockListIt> unmapped_free_blocks_;
  mutable SpinLock spinlock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
