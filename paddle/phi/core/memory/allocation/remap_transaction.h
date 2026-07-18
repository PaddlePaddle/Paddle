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

#if defined(PADDLE_WITH_CUDA)

#include <cstddef>
#include <functional>
#include <list>
#include <vector>

#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

namespace paddle {
namespace memory {
namespace allocation {

class RemapTransaction {
 public:
  using CommitDestinationAllocationsFn =
      std::function<void(std::vector<DecoratedAllocationPtr>*)>;
  using CanPrepareDestinationRangeFn = std::function<bool(void*, size_t)>;
  using PrepareDestinationRangeFn = std::function<bool(void*, size_t)>;
  using RollbackSourceMappingsFn = std::function<void()>;
  using SourcePages = std::vector<VMMBackingMap::MappedPage>;
  using BlockList = std::list<BlockV2>;
  using BlockIterator = BlockList::iterator;
  struct DestinationRollbackRange {
    VMMDevicePtr va{0};
    size_t handle_count{0};
  };
  struct DestinationBlockRollbackRange {
    VMMDevicePtr va{0};
    size_t size{0};
    bool is_tail{false};
  };
  struct SourceCollectionStats {
    size_t free_block_count{0};
    size_t safe_block_count{0};
    size_t fully_covered_count{0};
    size_t partial_count{0};
    size_t event_blocked_count{0};
    size_t fully_covered_bytes{0};
    size_t partial_bytes{0};
    size_t event_blocked_bytes{0};
    size_t unknown_safety_blocked_count{0};
    size_t unknown_safety_blocked_bytes{0};
    size_t remapped_blocked_count{0};
    size_t remapped_blocked_bytes{0};
  };
  struct DestinationPlacement {
    static DestinationPlacement AtTail(VMMDevicePtr dst, size_t count) {
      DestinationPlacement placement;
      placement.is_tail = true;
      placement.dst = dst;
      placement.count = count;
      return placement;
    }

    static DestinationPlacement InUnmappedRange(BlockIterator unmapped_it,
                                                VMMDevicePtr dst,
                                                size_t handle_start_idx,
                                                size_t count) {
      DestinationPlacement placement;
      placement.unmapped_it = unmapped_it;
      placement.dst = dst;
      placement.handle_start_idx = handle_start_idx;
      placement.count = count;
      return placement;
    }

    bool is_tail{false};
    BlockIterator unmapped_it;
    VMMDevicePtr dst{0};
    size_t handle_start_idx{0};
    size_t count{0};
  };
  enum class DestinationPlanKind {
    kNone,
    kTail,
    kSingleUnmappedRange,
    kScatterUnmappedRanges,
  };
  enum class DestinationPolicy {
    kTailThenAnyGap,
    kTailOnly,
    kDirectGapThenTail,
  };
  struct DestinationPlan {
    DestinationPlanKind kind{DestinationPlanKind::kNone};
    std::vector<DestinationPlacement> placements;

    bool HasPlacement() const { return kind != DestinationPlanKind::kNone; }
  };
  struct UnmappedDestinationPlan {
    BlockIterator single_it;
    std::vector<DestinationPlacement> scatter_placements;
    size_t total_capacity{0};
    size_t scatter_handles{0};
  };
  struct MoveCommitStats {
    uint64_t unmap_us{0};
    uint64_t map_us{0};
    uint64_t set_access_us{0};
    uint64_t metadata_us{0};
    uint64_t restore_us{0};
    uint64_t rollback_us{0};
    size_t unmap_ranges{0};
    size_t unmap_calls{0};
    size_t set_access_ranges{0};
    size_t set_access_calls{0};
  };
  struct MoveResult {
    bool success{false};
    bool used_tail{false};
    uint64_t destination_plan_us{0};
    uint64_t move_commit_us{0};
    MoveCommitStats move_stats;
    VMMDevicePtr target_min_va{0};
    VMMDevicePtr target_max_va{0};
  };
  struct PlannedSourceBlock {
    BlockIterator block_it;
    std::vector<BlockV2> replacement_segments;
  };
  struct SourceMovePlan {
    SourceCollectionStats stats;
    std::vector<VMMBackingMap::MappedPage> source_pages;
    std::vector<PlannedSourceBlock> source_blocks;
  };
  struct CompactResult {
    SourceCollectionStats source_stats;
    size_t remapped_handle_count{0};
    size_t remapped_bytes{0};
    uint64_t source_collect_us{0};
    uint64_t destination_plan_us{0};
    uint64_t move_commit_us{0};
    MoveCommitStats move_stats;
    VMMDevicePtr source_min_va{0};
    VMMDevicePtr source_max_va{0};
    VMMDevicePtr target_min_va{0};
    VMMDevicePtr target_max_va{0};
    bool success{false};
    bool used_tail{false};
  };
  RemapTransaction(
      CUDAVirtualMemAllocatorV2* vmm_allocator,
      size_t handle_size,
      CommitDestinationAllocationsFn commit_destination_allocations = {},
      CanPrepareDestinationRangeFn can_prepare_destination_range = {},
      PrepareDestinationRangeFn prepare_destination_range = {})
      : vmm_allocator_(vmm_allocator),
        handle_size_(handle_size),
        commit_destination_allocations_(
            std::move(commit_destination_allocations)),
        can_prepare_destination_range_(
            std::move(can_prepare_destination_range)),
        prepare_destination_range_(std::move(prepare_destination_range)) {}
  ~RemapTransaction();

  // Checks whether a mapped-free block is safe to remap without waiting. The
  // check may record a lazy event and remove completed pending states.
  static bool CheckBlockRemapSafety(BlockV2* block);

  BlockV2 MaterializeDestinationRange(VMMDevicePtr dst,
                                      const SourcePages& source_pages,
                                      size_t start,
                                      size_t count,
                                      PoolType pool_type);
  BlockV2 MaterializeDestinationPlacement(const DestinationPlacement& placement,
                                          const SourcePages& source_pages,
                                          PoolType pool_type);
  bool CollectTargetPagesForRange(
      VMMDevicePtr dst,
      size_t handle_count,
      const char* context,
      std::vector<VMMBackingMap::UnmappedPage>* target_pages) const;
  bool PrepareDestinationPlacement(
      const DestinationPlacement& placement,
      const char* context,
      std::vector<VMMBackingMap::UnmappedPage>* target_pages) const;
  bool CanPrepareDestinationRange(VMMDevicePtr dst, size_t size) const;
  bool PrepareDestinationRange(VMMDevicePtr dst,
                               size_t size,
                               const char* context) const;
  SourceMovePlan CollectRemapSourcePlan(BlockList* blocks,
                                        size_t requested_size,
                                        PoolType pool_type,
                                        const SourcePages& source_pages);
  void ApplyPlannedSourceBlocks(
      BlockList* blocks, std::vector<PlannedSourceBlock>* source_blocks) const;
  bool TailIsUsable(VMMDevicePtr tail_va,
                    size_t total_bytes,
                    VMMDevicePtr va_limit) const;
  size_t CountLeadingUnmappedBackingPages(VMMDevicePtr va, size_t size) const;
  UnmappedDestinationPlan PlanUnmappedDestinations(BlockList* blocks,
                                                   size_t handle_count) const;
  DestinationPlan SelectDestinationPlan(BlockList* blocks,
                                        VMMDevicePtr tail_va,
                                        VMMDevicePtr va_limit,
                                        size_t handle_count,
                                        DestinationPolicy policy) const;
  bool TryMoveToTail(BlockList* blocks,
                     VMMDevicePtr tail_va,
                     SourceMovePlan* plan,
                     PoolType pool_type,
                     MoveCommitStats* move_stats);
  bool MovePlannedPagesToTargets(
      BlockList* blocks,
      SourceMovePlan* plan,
      const std::vector<VMMBackingMap::UnmappedPage>& target_pages,
      MoveCommitStats* move_stats);
  bool TryMoveToUnmappedRange(BlockList* blocks,
                              BlockIterator unmapped_it,
                              SourceMovePlan* plan,
                              PoolType pool_type,
                              MoveCommitStats* move_stats);
  bool TryMoveToUnmappedRanges(
      BlockList* blocks,
      SourceMovePlan* plan,
      const std::vector<DestinationPlacement>& placements,
      PoolType pool_type,
      MoveCommitStats* move_stats);
  MoveResult ExecuteDestinationPlan(BlockList* blocks,
                                    VMMDevicePtr tail_va,
                                    VMMDevicePtr va_limit,
                                    SourceMovePlan* plan,
                                    PoolType pool_type,
                                    DestinationPolicy policy);
  CompactResult CompactFreeBlocks(
      BlockList* blocks,
      size_t requested_size,
      PoolType pool_type,
      const SourcePages& source_pages,
      DestinationPolicy policy = DestinationPolicy::kTailThenAnyGap);
  void InstallTailFreeBlock(BlockList* blocks, BlockV2 free_block) const;
  BlockIterator ReplaceUnmappedRangeWithMappedFree(BlockList* blocks,
                                                   BlockIterator unmapped_it,
                                                   BlockV2 mapped_free_block,
                                                   PoolType pool_type) const;
  void InstallDestination(BlockList* blocks,
                          const DestinationPlacement& placement,
                          BlockV2 mapped_free_block,
                          PoolType pool_type);
  void MergeAdjacentFreeBlocks(BlockList* blocks) const;
  void MergeAdjacentUnmappedFreeBlocks(BlockList* blocks) const;
  void NormalizeBlocks(BlockList* blocks) const;
  bool RestoreRangeAsMappedFree(BlockList* blocks,
                                VMMDevicePtr va,
                                size_t size);
  bool ReplaceRangeWithUnmappedFree(BlockList* blocks,
                                    VMMDevicePtr va,
                                    size_t size);
  bool RemoveTailDestinationBlock(BlockList* blocks,
                                  VMMDevicePtr va,
                                  size_t size);
  void RestoreRemappedSourcesToFreeBlocks(BlockList* blocks,
                                          const SourcePages& source_pages);
  void Commit();
  void Rollback();

 private:
  // Record successfully mapped destination ranges so later bookkeeping
  // failures can still unmap every destination owned by this transaction.
  void RecordDestinationRollbackRange(VMMDevicePtr dst, size_t handle_count);
  void RecordDestinationBlockRollbackRange(
      const DestinationPlacement& placement);
  // Stop transaction rollback from unmapping a destination after its cleanup
  // ownership has been transferred to the underlying allocation registry.
  void DiscardDestinationRollbackRange(VMMDevicePtr dst, size_t size);
  void RollbackDestinations();
  void RollbackDestinationBlockViews();
  void StageDestinationAllocation(Allocation* allocation);
  bool HasPendingState() const;

  CUDAVirtualMemAllocatorV2* vmm_allocator_;
  size_t handle_size_;
  CommitDestinationAllocationsFn commit_destination_allocations_;
  CanPrepareDestinationRangeFn can_prepare_destination_range_;
  PrepareDestinationRangeFn prepare_destination_range_;
  std::vector<DestinationRollbackRange> destination_rollback_ranges_;
  std::vector<DestinationBlockRollbackRange> destination_block_rollback_ranges_;
  RollbackSourceMappingsFn rollback_source_mappings_;
  std::vector<Allocation*> pending_destination_allocations_;
  BlockList* blocks_{nullptr};
  bool completed_{false};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
