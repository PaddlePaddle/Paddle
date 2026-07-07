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
  using CommitSyntheticAllocationFn =
      std::function<void(DecoratedAllocationPtr)>;
  using CanUseDestinationRangeFn = std::function<bool(void*, size_t)>;
  using ReleaseStaleDestinationAllocationsFn =
      std::function<bool(void*, size_t)>;
  using RollbackMappedDestinationFn = std::function<void()>;
  using RollbackSourceMappingsFn = std::function<void()>;
  using VaRanges = std::vector<std::pair<VMMDevicePtr, size_t>>;
  using BlockList = std::list<BlockV2>;
  using BlockIterator = BlockList::iterator;
  struct MaterializedRange {
    BlockV2 free_block;
    size_t bytes{0};
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
    static DestinationPlacement Tail(VMMDevicePtr dst, size_t count) {
      DestinationPlacement placement;
      placement.is_tail = true;
      placement.dst = dst;
      placement.count = count;
      return placement;
    }

    static DestinationPlacement UnmappedFree(BlockIterator unmapped_free_it,
                                             VMMDevicePtr dst,
                                             size_t handle_start_idx,
                                             size_t count) {
      DestinationPlacement placement;
      placement.unmapped_free_it = unmapped_free_it;
      placement.dst = dst;
      placement.handle_start_idx = handle_start_idx;
      placement.count = count;
      return placement;
    }

    bool is_tail{false};
    BlockIterator unmapped_free_it;
    VMMDevicePtr dst{0};
    size_t handle_start_idx{0};
    size_t count{0};
  };
  enum class DestinationPlanKind {
    kNone,
    kTail,
    kSingleUnmappedFree,
    kScatterUnmappedFree,
  };
  enum class DestinationPlanFailure {
    kNone,
    kInsufficientUnmappedFreeCapacity,
    kScatterPlanFailed,
  };
  struct DestinationPlan {
    DestinationPlanKind kind{DestinationPlanKind::kNone};
    DestinationPlanFailure failure{DestinationPlanFailure::kNone};
    std::vector<DestinationPlacement> placements;

    bool HasPlacement() const { return kind != DestinationPlanKind::kNone; }
  };
  struct UnmappedFreeDestinationPlan {
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
  struct PlacementResult {
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
    std::vector<VMMAllocHandle> handles;
    std::vector<std::shared_ptr<VMMHandleMeta>> metas;
    std::vector<VMMBackingMap::MappedPage> source_pages;
    std::vector<PlannedSourceBlock> source_blocks;
  };
  struct PreScanResult {
    size_t free_range_count{0};
    size_t unmapped_free_range_count{0};
    size_t mapped_page_count{0};
    size_t target_page_count{0};
    bool source_ok{true};
    bool target_ok{true};
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
  RemapTransaction(CUDAVirtualMemAllocatorV2* vmm_allocator,
                   size_t handle_size,
                   CommitSyntheticAllocationFn commit_synthetic_allocation = {},
                   CanUseDestinationRangeFn can_use_destination_range = {},
                   ReleaseStaleDestinationAllocationsFn
                       release_stale_destination_allocations = {})
      : vmm_allocator_(vmm_allocator),
        handle_size_(handle_size),
        commit_synthetic_allocation_(std::move(commit_synthetic_allocation)),
        can_use_destination_range_(std::move(can_use_destination_range)),
        release_stale_destination_allocations_(
            std::move(release_stale_destination_allocations)) {}
  ~RemapTransaction();

  PreScanResult PreparePhase1Diagnostics(BlockList* blocks,
                                         size_t requested_size,
                                         const char* source_context,
                                         const char* target_context);
  const VMMBackingMap::CompactCandidates& candidates() const {
    return candidates_;
  }

  MaterializedRange MaterializeMappedRange(
      VMMDevicePtr dst,
      const std::vector<VMMAllocHandle>& handles,
      size_t start,
      size_t count,
      PoolType pool_type);
  MaterializedRange MaterializeDestinationPlacement(
      const DestinationPlacement& placement,
      const std::vector<VMMAllocHandle>& handles,
      PoolType pool_type);
  bool CollectTargetPagesForRange(
      VMMDevicePtr dst,
      size_t handle_count,
      const char* context,
      std::vector<VMMBackingMap::UnmappedPage>* target_pages) const;
  bool PrepareMoveDestinationPlacement(
      const DestinationPlacement& placement,
      const char* context,
      std::vector<VMMBackingMap::UnmappedPage>* target_pages) const;
  bool CanUseDestinationRange(VMMDevicePtr dst, size_t size) const;
  bool PrepareDestinationRange(VMMDevicePtr dst,
                               size_t size,
                               const char* context) const;
  bool ReleaseStaleDestinationAllocations(VMMDevicePtr dst, size_t size) const;
  SourceMovePlan CollectRemapSourcePlan(BlockList* blocks,
                                        size_t requested_size,
                                        PoolType pool_type);
  void ApplyPlannedSourceBlocks(
      BlockList* blocks, std::vector<PlannedSourceBlock>* source_blocks) const;
  bool TailIsUsable(VMMDevicePtr tail_va,
                    size_t total_bytes,
                    VMMDevicePtr va_limit) const;
  size_t CountLeadingUnmappedBackingPages(VMMDevicePtr va, size_t size) const;
  UnmappedFreeDestinationPlan PlanUnmappedFreeDestinations(
      BlockList* blocks, size_t handle_count) const;
  DestinationPlan SelectDestinationPlan(BlockList* blocks,
                                        VMMDevicePtr tail_va,
                                        VMMDevicePtr va_limit,
                                        size_t handle_count,
                                        const char* log_prefix) const;
  bool TryCommitTailMovePlacement(BlockList* blocks,
                                  VMMDevicePtr tail_va,
                                  SourceMovePlan* plan,
                                  PoolType pool_type,
                                  MoveCommitStats* move_stats);
  bool MovePlannedPagesToTargets(
      BlockList* blocks,
      SourceMovePlan* plan,
      const std::vector<VMMBackingMap::UnmappedPage>& target_pages,
      MoveCommitStats* move_stats);
  bool TryCommitSingleUnmappedFreeMovePlacement(BlockList* blocks,
                                                BlockIterator unmapped_free_it,
                                                SourceMovePlan* plan,
                                                PoolType pool_type,
                                                MoveCommitStats* move_stats);
  bool TryCommitUnmappedFreeMoveScatter(
      BlockList* blocks,
      SourceMovePlan* plan,
      const std::vector<DestinationPlacement>& placements,
      PoolType pool_type,
      MoveCommitStats* move_stats);
  PlacementResult ExecuteMovePlacementStrategy(BlockList* blocks,
                                               VMMDevicePtr tail_va,
                                               VMMDevicePtr va_limit,
                                               SourceMovePlan* plan,
                                               PoolType pool_type);
  CompactResult CompactFreeBlocks(BlockList* blocks,
                                  size_t requested_size,
                                  PoolType pool_type);
  void InstallTailFreeBlock(BlockList* blocks, BlockV2 free_block) const;
  BlockIterator InstallMappedUnmappedFreeRange(BlockList* blocks,
                                               BlockIterator unmapped_free_it,
                                               BlockV2 free_block,
                                               PoolType pool_type) const;
  void InstallMappedDestinationRange(BlockList* blocks,
                                     const DestinationPlacement& placement,
                                     BlockV2 free_block,
                                     PoolType pool_type) const;
  BlockV2 MakeUnmappedFreeBlock(void* ptr,
                                size_t size,
                                PoolType pool_type) const;
  void MergeAdjacentFreeBlocks(BlockList* blocks) const;
  void MergeAdjacentUnmappedFreeBlocks(BlockList* blocks) const;
  void NormalizeBlocks(BlockList* blocks) const;
  bool RestoreUnmappedFreeRangeToMappedFreeBlock(
      BlockList* blocks,
      VMMDevicePtr va,
      size_t size,
      const std::shared_ptr<VMMHandleMeta>& meta);
  bool RestoreRemappedSourcesToFreeBlocks(
      BlockList* blocks,
      const std::vector<VMMAllocHandle>& handles,
      const std::vector<std::shared_ptr<VMMHandleMeta>>& metas);
  void Commit();
  void Rollback();

 private:
  // Record successfully mapped destination ranges so later bookkeeping
  // failures can still unmap every destination owned by this transaction.
  void RecordMappedDestinationRange(VMMDevicePtr dst, size_t handle_count);
  void RollbackMappedDestinations();
  void StageSyntheticAllocation(Allocation* allocation);
  bool HasPendingState() const;

  CUDAVirtualMemAllocatorV2* vmm_allocator_;
  size_t handle_size_;
  VMMBackingMap::CompactCandidates candidates_;
  CommitSyntheticAllocationFn commit_synthetic_allocation_;
  CanUseDestinationRangeFn can_use_destination_range_;
  ReleaseStaleDestinationAllocationsFn release_stale_destination_allocations_;
  std::vector<RollbackMappedDestinationFn> rollback_mapped_destinations_;
  RollbackSourceMappingsFn rollback_source_mappings_;
  std::vector<Allocation*> pending_synthetic_allocations_;
  bool completed_{false};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
