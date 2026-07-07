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

#include "paddle/phi/core/memory/allocation/remap_transaction.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <list>
#include <unordered_map>
#include <utility>

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kRemapSourceUnmapChunkSize = 64UL << 20;

uint64_t ElapsedMicros(Clock::time_point start, Clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());
}

size_t RemapSourceUnmapChunkHandles(size_t handle_size) {
  if (handle_size == 0) {
    return 1;
  }
  return std::max<size_t>(1, kRemapSourceUnmapChunkSize / handle_size);
}

std::vector<std::pair<VMMDevicePtr, size_t>> CollectFreeRanges(
    const std::list<BlockV2>& blocks) {
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges;
  for (const auto& block : blocks) {
    if (!block.IsMappedFree()) {
      continue;
    }
    ranges.emplace_back(block.va_range());
  }
  return ranges;
}

std::vector<std::pair<VMMDevicePtr, size_t>> CollectUnmappedFreeRanges(
    const std::list<BlockV2>& blocks) {
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges;
  for (const auto& block : blocks) {
    if (!block.IsUnmappedFree()) {
      continue;
    }
    ranges.emplace_back(block.va_range());
  }
  return ranges;
}

using RemapSourceState = VMMBackingMap::RemapSourceState;

VMMDevicePtr AlignUp(VMMDevicePtr value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const auto remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

bool QueryRemapEvent(VMMBlockRemapState* state) {
  if (state->event == nullptr) {
    return true;
  }
  auto err = cudaEventQuery(state->event->event);
  if (err == cudaErrorNotReady) {
    return false;
  }
  PADDLE_ENFORCE_GPU_SUCCESS(err);
  state->event.reset();
  state->stream = nullptr;
  return true;
}

bool RecordRemapEvent(VMMBlockRemapState* state) {
  if (state->stream == nullptr || state->event != nullptr) {
    return true;
  }
  gpuEvent_t event = nullptr;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, state->stream));
  state->event = std::make_shared<CUDAEventGuard>(event);
  return false;
}

bool RemapStateReady(VMMBlockRemapState* state) {
  if (!QueryRemapEvent(state)) {
    return false;
  }
  if (state->stream == nullptr) {
    return true;
  }
  auto err = cudaStreamQuery(state->stream);
  if (err == cudaSuccess) {
    state->stream = nullptr;
    state->event.reset();
    return true;
  }
  if (err != cudaErrorNotReady) {
    PADDLE_ENFORCE_GPU_SUCCESS(err);
  }
  return RecordRemapEvent(state) && QueryRemapEvent(state);
}

bool IsRemapSafe(BlockV2* block) {
  if (!block->IsMappedFree()) {
    return false;
  }
  if (block->HasUnknownRemapSafety()) {
    return false;
  }
  bool ready = true;
  VMMBlockRemapState primary{block->owning_stream_, block->remap_safe_event_};
  if (!RemapStateReady(&primary)) {
    ready = false;
  }
  block->owning_stream_ = primary.stream;
  block->remap_safe_event_ = std::move(primary.event);

  for (auto it = block->remap_pending_states_.begin();
       it != block->remap_pending_states_.end();) {
    if (!RemapStateReady(&*it)) {
      ready = false;
      ++it;
      continue;
    }
    if (it->stream == nullptr && it->event == nullptr) {
      it = block->remap_pending_states_.erase(it);
    } else {
      ++it;
    }
  }
  return ready;
}

void AppendMappedFreeSubRange(std::vector<BlockV2>* segments,
                              const BlockV2& source,
                              VMMDevicePtr va,
                              size_t size) {
  if (size == 0) {
    return;
  }

  BlockV2 segment = source.MakeMappedFreeSubBlock(va - source.begin_va(), size);
  if (!segments->empty() &&
      segments->back().CanMergeAdjacentFreeBlock(segment)) {
    segments->back().MergeAdjacentBlock(segment);
    return;
  }
  segments->push_back(std::move(segment));
}

void AppendUnmappedFreeRange(std::vector<BlockV2>* segments,
                             VMMDevicePtr va,
                             size_t size,
                             PoolType pool_type) {
  if (size == 0) {
    return;
  }

  BlockV2 segment = BlockV2::MakeUnmappedFreeBlock(
      reinterpret_cast<void*>(va), size, pool_type);
  if (!segments->empty() &&
      segments->back().CanMergeAdjacentUnmappedFreeBlock(segment)) {
    segments->back().MergeAdjacentUnmappedFreeBlock(segment);
    return;
  }
  segments->push_back(std::move(segment));
}

using SourcePageMap =
    std::unordered_map<VMMDevicePtr, VMMBackingMap::MappedPage>;

SourcePageMap CollectSourcePageCandidates(
    CUDAVirtualMemAllocatorV2* vmm_allocator,
    const std::list<BlockV2>& blocks,
    size_t requested_size) {
  auto source_ranges = CollectFreeRanges(blocks);
  auto source_pages =
      vmm_allocator->CollectRemapSourcePages(source_ranges, requested_size);
  SourcePageMap candidates;
  candidates.reserve(source_pages.size());
  for (const auto& page : source_pages) {
    candidates.emplace(page.va, page);
  }
  return candidates;
}

void RestoreSourceMappings(
    std::list<BlockV2>* blocks,
    const std::vector<VMMAllocHandle>& handles,
    const std::vector<std::shared_ptr<VMMHandleMeta>>& metas,
    CUDAVirtualMemAllocatorV2* vmm_allocator,
    size_t handle_size,
    const std::function<void(std::list<BlockV2>*)>& merge_adjacent_free_blocks,
    const std::function<bool(std::list<BlockV2>*,
                             VMMDevicePtr,
                             size_t,
                             const std::shared_ptr<VMMHandleMeta>&)>&
        restore_unmapped_free_to_mapped_free) {
  size_t restored = 0, force_released = 0;
  for (size_t i = 0; i < handles.size(); ++i) {
    auto restore_result = vmm_allocator->RestoreRemapSourceMapping(
        handles[i], metas[i], handle_size);
    if (restore_result ==
        CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult::kSkipped) {
      continue;
    }
    if (restore_result ==
        CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult::kForceReleased) {
      force_released++;
      continue;
    }

    VMMDevicePtr original_va = metas[i]->base();
    if (restore_unmapped_free_to_mapped_free(
            blocks, original_va, handle_size, metas[i])) {
      metas[i]->RestoreOriginalOwnership();
      restored++;
    } else {
      vmm_allocator->ForceReleaseRestoredRemapSourceMapping(
          handles[i], metas[i], handle_size, "block restore", true);
      force_released++;
    }
  }
  merge_adjacent_free_blocks(blocks);
  VLOG(3) << "RestoreSourceMappings: restored=" << restored
          << " force_released=" << force_released;
}

}  // namespace

RemapTransaction::~RemapTransaction() {
  if (!completed_ && HasPendingState()) {
    VLOG(3) << "VMM V2 remap transaction destroyed before Commit/Rollback; "
               "rolling back pending state";
    Rollback();
  }
}

RemapTransaction::PreScanResult RemapTransaction::PreparePhase1Diagnostics(
    BlockList* blocks,
    size_t requested_size,
    const char* source_context,
    const char* target_context) {
  PreScanResult result;
  auto free_ranges = CollectFreeRanges(*blocks);
  auto unmapped_free_ranges = CollectUnmappedFreeRanges(*blocks);
  result.free_range_count = free_ranges.size();
  result.unmapped_free_range_count = unmapped_free_ranges.size();
  candidates_ = vmm_allocator_->CollectCompactCandidates(
      free_ranges, unmapped_free_ranges, requested_size);
  result.mapped_page_count = candidates_.source_pages.size();
  result.target_page_count = candidates_.target_pages.size();
  result.source_ok = vmm_allocator_->ValidateMappedPages(
      candidates_.source_pages, source_context);
  result.target_ok = vmm_allocator_->ValidateUnmappedPages(
      candidates_.target_pages, target_context);
  return result;
}

RemapTransaction::MaterializedRange RemapTransaction::MaterializeMappedRange(
    VMMDevicePtr dst,
    const std::vector<VMMAllocHandle>& handles,
    size_t start,
    size_t count,
    PoolType pool_type) {
  CUDAVirtualMemAllocatorV2::StagedRemapDestination staged;
  try {
    staged = vmm_allocator_->CreateStagedRemapDestination(
        dst, handles, start, count, pool_type);
  } catch (const std::exception& e) {
    VLOG(3) << "VMM V2 remap transaction: materialize mapped range failed, "
               "dst="
            << reinterpret_cast<void*>(dst) << " start=" << start
            << " count=" << count << " total_handles=" << handles.size()
            << " error=" << e.what();
    throw;
  } catch (...) {
    VLOG(3) << "VMM V2 remap transaction: materialize mapped range failed "
               "with unknown exception, dst="
            << reinterpret_cast<void*>(dst) << " start=" << start
            << " count=" << count << " total_handles=" << handles.size();
    throw;
  }
  try {
    StageSyntheticAllocation(staged.allocation);
  } catch (const std::exception& e) {
    VLOG(3) << "VMM V2 remap transaction: failed to stage synthetic "
               "allocation for rollback, destroying it. dst="
            << reinterpret_cast<void*>(dst) << " bytes=" << staged.bytes
            << " error=" << e.what();
    vmm_allocator_->DestroyStagedSyntheticAllocation(staged.allocation);
    staged.allocation = nullptr;
    throw;
  } catch (...) {
    VLOG(3) << "VMM V2 remap transaction: unknown failure while staging "
               "synthetic allocation for rollback, destroying it. dst="
            << reinterpret_cast<void*>(dst) << " bytes=" << staged.bytes;
    vmm_allocator_->DestroyStagedSyntheticAllocation(staged.allocation);
    staged.allocation = nullptr;
    throw;
  }
  MaterializedRange range;
  range.bytes = staged.bytes;
  range.free_block = std::move(staged.block);
  return range;
}

RemapTransaction::MaterializedRange
RemapTransaction::MaterializeDestinationPlacement(
    const DestinationPlacement& placement,
    const std::vector<VMMAllocHandle>& handles,
    PoolType pool_type) {
  return MaterializeMappedRange(placement.dst,
                                handles,
                                placement.handle_start_idx,
                                placement.count,
                                pool_type);
}

bool RemapTransaction::CollectTargetPagesForRange(
    VMMDevicePtr dst,
    size_t handle_count,
    const char* context,
    std::vector<VMMBackingMap::UnmappedPage>* target_pages) const {
  const size_t bytes = handle_count * handle_size_;
  std::vector<std::pair<VMMDevicePtr, size_t>> target_ranges = {{dst, bytes}};
  *target_pages = vmm_allocator_->CollectUnmappedPages(target_ranges, bytes);
  if (target_pages->size() != handle_count ||
      !vmm_allocator_->ValidateUnmappedPages(*target_pages, context)) {
    VLOG(3) << "VMM V2 remap transaction: target validation failed in "
            << context << ", dst=" << reinterpret_cast<void*>(dst)
            << " target_pages=" << target_pages->size()
            << " handles=" << handle_count;
    return false;
  }
  return true;
}

bool RemapTransaction::PrepareMoveDestinationPlacement(
    const DestinationPlacement& placement,
    const char* context,
    std::vector<VMMBackingMap::UnmappedPage>* target_pages) const {
  if (!CollectTargetPagesForRange(
          placement.dst, placement.count, context, target_pages)) {
    return false;
  }
  return PrepareDestinationRange(
      placement.dst, placement.count * handle_size_, context);
}

bool RemapTransaction::CanUseDestinationRange(VMMDevicePtr dst,
                                              size_t size) const {
  if (!can_use_destination_range_) {
    return true;
  }
  return can_use_destination_range_(reinterpret_cast<void*>(dst), size);
}

bool RemapTransaction::PrepareDestinationRange(VMMDevicePtr dst,
                                               size_t size,
                                               const char* context) const {
  if (ReleaseStaleDestinationAllocations(dst, size)) {
    return true;
  }
  VLOG(3) << "VMM V2 remap transaction: synthetic allocation preparation "
             "failed in "
          << context << ", dst=" << reinterpret_cast<void*>(dst)
          << " bytes=" << size;
  return false;
}

bool RemapTransaction::ReleaseStaleDestinationAllocations(VMMDevicePtr dst,
                                                          size_t size) const {
  if (!release_stale_destination_allocations_) {
    return true;
  }
  return release_stale_destination_allocations_(reinterpret_cast<void*>(dst),
                                                size);
}

RemapTransaction::SourceMovePlan RemapTransaction::CollectRemapSourcePlan(
    BlockList* blocks, size_t requested_size, PoolType pool_type) {
  SourceMovePlan plan;
  bool logged_first_candidate = false;
  auto source_candidates =
      CollectSourcePageCandidates(vmm_allocator_, *blocks, requested_size);
  for (auto it = blocks->begin(); it != blocks->end();) {
    auto current = it++;
    if (current->IsFree()) plan.stats.free_block_count++;
    if (!current->IsMappedFree()) {
      continue;
    }
    if (current->HasUnknownRemapSafety()) {
      plan.stats.unknown_safety_blocked_count +=
          (current->size() + handle_size_ - 1) / handle_size_;
      plan.stats.unknown_safety_blocked_bytes += current->size();
      continue;
    }
    if (!IsRemapSafe(&*current)) {
      plan.stats.event_blocked_count +=
          (current->size() + handle_size_ - 1) / handle_size_;
      plan.stats.event_blocked_bytes += current->size();
      continue;
    }
    plan.stats.safe_block_count++;

    std::vector<BlockV2> replacement_segments;
    size_t remapped_count_before = plan.handles.size();
    VMMDevicePtr cursor = current->begin_va();
    const VMMDevicePtr block_end = current->end_va();
    for (VMMDevicePtr page_va = AlignUp(current->begin_va(), handle_size_);
         page_va + handle_size_ <= block_end;
         page_va += handle_size_) {
      auto candidate_it = source_candidates.find(page_va);
      if (candidate_it == source_candidates.end()) {
        plan.stats.partial_count++;
        plan.stats.partial_bytes += handle_size_;
        continue;
      }

      const auto& candidate = candidate_it->second;
      const auto source_state = candidate.remap_source_state;
      if (cursor < page_va) {
        AppendMappedFreeSubRange(
            &replacement_segments, *current, cursor, page_va - cursor);
      }
      if (source_state == RemapSourceState::kReady) {
        plan.stats.fully_covered_count++;
        plan.stats.fully_covered_bytes += handle_size_;
        if (!logged_first_candidate) {
          VLOG(4) << "First move-page candidate pool="
                  << static_cast<int>(pool_type)
                  << " block_ptr=" << current->ptr()
                  << " block_size=" << current->size()
                  << " handle_base=" << reinterpret_cast<void*>(candidate.va)
                  << " handle_size=" << handle_size_
                  << " handle=" << reinterpret_cast<void*>(candidate.handle);
          logged_first_candidate = true;
        }
        plan.source_pages.push_back(candidate);
        plan.handles.push_back(candidate.handle);
        plan.metas.push_back(candidate.meta);
        AppendUnmappedFreeRange(
            &replacement_segments, page_va, handle_size_, pool_type);
        cursor = page_va + handle_size_;
        continue;
      }
      switch (source_state) {
        case RemapSourceState::kRemapDestinationOwned:
          plan.stats.remapped_blocked_count++;
          plan.stats.remapped_blocked_bytes += handle_size_;
          break;
        case RemapSourceState::kPendingEvent:
          plan.stats.event_blocked_count++;
          plan.stats.event_blocked_bytes += handle_size_;
          break;
        case RemapSourceState::kPartialOrInvalid:
          plan.stats.partial_count++;
          plan.stats.partial_bytes += handle_size_;
          break;
        case RemapSourceState::kReady:
          break;
      }
      AppendMappedFreeSubRange(
          &replacement_segments, *current, page_va, handle_size_);
      cursor = page_va + handle_size_;
    }
    if (cursor < block_end) {
      AppendMappedFreeSubRange(
          &replacement_segments, *current, cursor, block_end - cursor);
    }

    if (plan.handles.size() == remapped_count_before) {
      continue;
    }
    plan.source_blocks.push_back({current, std::move(replacement_segments)});

    if (requested_size > 0 &&
        plan.handles.size() * handle_size_ >= requested_size) {
      VLOG(3) << "VMM V2 remap transaction: bounded move-page"
              << " exit, collected " << plan.handles.size() << " handles ("
              << plan.handles.size() * handle_size_
              << " bytes) >= requested=" << requested_size;
      break;
    }
  }
  return plan;
}

void RemapTransaction::ApplyPlannedSourceBlocks(
    BlockList* blocks, std::vector<PlannedSourceBlock>* source_blocks) const {
  for (auto& source_block : *source_blocks) {
    auto insert_pos = source_block.block_it;
    for (auto& segment : source_block.replacement_segments) {
      blocks->insert(insert_pos, std::move(segment));
    }
    blocks->erase(source_block.block_it);
  }
  source_blocks->clear();
}

bool RemapTransaction::TailIsUsable(VMMDevicePtr tail_va,
                                    size_t total_bytes,
                                    VMMDevicePtr va_limit) const {
  if (tail_va + total_bytes > va_limit) {
    return false;
  }
  return vmm_allocator_->IsRangeUnmapped(tail_va, total_bytes);
}

size_t RemapTransaction::CountLeadingUnmappedBackingPages(VMMDevicePtr va,
                                                          size_t size) const {
  const size_t aligned_size = (size / handle_size_) * handle_size_;
  if (aligned_size == 0) {
    return 0;
  }
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {{va, aligned_size}};
  auto pages = vmm_allocator_->CollectUnmappedPages(ranges, aligned_size);
  size_t leading = 0;
  for (const auto& page : pages) {
    if (page.va != va + leading * handle_size_) {
      break;
    }
    ++leading;
  }
  return leading;
}

RemapTransaction::UnmappedFreeDestinationPlan
RemapTransaction::PlanUnmappedFreeDestinations(BlockList* blocks,
                                               size_t handle_count) const {
  UnmappedFreeDestinationPlan plan;
  plan.single_it = blocks->end();
  const size_t required_bytes = handle_count * handle_size_;
  const size_t unknown_capacity = static_cast<size_t>(-1);
  std::vector<std::pair<BlockIterator, size_t>> candidates;
  for (auto it = blocks->begin(); it != blocks->end(); ++it) {
    if (!it->IsUnmappedFree()) continue;
    size_t unmapped_free_cap = unknown_capacity;
    if (it->size() >= required_bytes) {
      unmapped_free_cap =
          CountLeadingUnmappedBackingPages(it->begin_va(), it->size());
      const size_t capacity_bytes = unmapped_free_cap * handle_size_;
      if (capacity_bytes >= required_bytes &&
          CanUseDestinationRange(it->begin_va(), required_bytes)) {
        plan.single_it = it;
        return plan;
      }
    }
    candidates.push_back({it, unmapped_free_cap});
  }

  size_t handle_idx = 0;
  for (const auto& candidate : candidates) {
    auto it = candidate.first;
    size_t unmapped_free_cap = candidate.second;
    if (unmapped_free_cap == unknown_capacity) {
      unmapped_free_cap =
          CountLeadingUnmappedBackingPages(it->begin_va(), it->size());
    }
    if (unmapped_free_cap == 0) continue;

    auto dst = it->begin_va();
    const size_t capacity_bytes = unmapped_free_cap * handle_size_;
    if (CanUseDestinationRange(dst, capacity_bytes)) {
      plan.total_capacity += capacity_bytes;
    }

    if (handle_idx >= handle_count) continue;
    size_t to_fill = std::min(unmapped_free_cap, handle_count - handle_idx);
    if (!CanUseDestinationRange(dst, to_fill * handle_size_)) {
      continue;
    }
    plan.scatter_placements.push_back(
        DestinationPlacement::UnmappedFree(it, dst, handle_idx, to_fill));
    handle_idx += to_fill;
  }
  plan.scatter_handles = handle_idx;
  return plan;
}

RemapTransaction::DestinationPlan RemapTransaction::SelectDestinationPlan(
    BlockList* blocks,
    VMMDevicePtr tail_va,
    VMMDevicePtr va_limit,
    size_t handle_count,
    const char* log_prefix) const {
  DestinationPlan plan;
  const size_t total_remapped = handle_count * handle_size_;

  const bool tail_driver_va_usable =
      TailIsUsable(tail_va, total_remapped, va_limit);
  if (tail_driver_va_usable &&
      CanUseDestinationRange(tail_va, total_remapped)) {
    VLOG(10) << "VMM remap compact using " << log_prefix
             << " tail path, dst_va=" << reinterpret_cast<void*>(tail_va)
             << " bytes=" << total_remapped;
    plan.kind = DestinationPlanKind::kTail;
    plan.placements.push_back(
        DestinationPlacement::Tail(tail_va, handle_count));
    return plan;
  }
  if (tail_driver_va_usable) {
    VLOG(3) << "VMM V2 remap transaction: " << log_prefix
            << " tail driver VA is usable but synthetic preparation rejected, "
            << "trying unmapped-free destination";
  }

  auto unmapped_free_plan = PlanUnmappedFreeDestinations(blocks, handle_count);
  if (unmapped_free_plan.single_it != blocks->end()) {
    BlockIterator unmapped_free_it = unmapped_free_plan.single_it;
    const VMMDevicePtr unmapped_free_va = unmapped_free_it->begin_va();
    VLOG(10) << "VMM remap compact using " << log_prefix
             << " unmapped-free path, dst_va="
             << reinterpret_cast<void*>(unmapped_free_va)
             << " unmapped_free_size=" << unmapped_free_it->size()
             << " bytes=" << total_remapped;
    plan.kind = DestinationPlanKind::kSingleUnmappedFree;
    plan.placements.push_back(DestinationPlacement::UnmappedFree(
        unmapped_free_it, unmapped_free_va, 0, handle_count));
    return plan;
  }

  VLOG(3) << "VMM V2 remap transaction: " << log_prefix
          << " tail unavailable and no single unmapped-free block >= "
          << total_remapped << " bytes, trying scatter";
  plan.kind = DestinationPlanKind::kScatterUnmappedFree;
  plan.placements = std::move(unmapped_free_plan.scatter_placements);
  if (unmapped_free_plan.scatter_handles == handle_count) {
    return plan;
  }
  if (unmapped_free_plan.total_capacity < total_remapped) {
    VLOG(3) << "VMM V2 remap transaction: " << log_prefix
            << " unmapped-free capacity " << unmapped_free_plan.total_capacity
            << " < total_remapped " << total_remapped;
    plan.kind = DestinationPlanKind::kNone;
    plan.failure = DestinationPlanFailure::kInsufficientUnmappedFreeCapacity;
    return plan;
  }

  size_t planned_handles = 0;
  for (const auto& p : plan.placements) {
    planned_handles += p.count;
  }
  VLOG(3) << "VMM V2 remap transaction " << log_prefix << " scatter: placed "
          << planned_handles << " of " << handle_count
          << " handles despite precheck";
  plan.kind = DestinationPlanKind::kNone;
  plan.failure = DestinationPlanFailure::kScatterPlanFailed;
  return plan;
}

bool RemapTransaction::TryCommitTailMovePlacement(BlockList* blocks,
                                                  VMMDevicePtr tail_va,
                                                  SourceMovePlan* plan,
                                                  PoolType pool_type,
                                                  MoveCommitStats* move_stats) {
  const size_t handle_count = plan->handles.size();
  auto placement = DestinationPlacement::Tail(tail_va, handle_count);
  std::vector<VMMBackingMap::UnmappedPage> target_pages;
  if (!PrepareMoveDestinationPlacement(
          placement,
          "RemapTransaction::TryCommitTailMovePlacement",
          &target_pages)) {
    return false;
  }

  if (!MovePlannedPagesToTargets(blocks, plan, target_pages, move_stats)) {
    return false;
  }
  ApplyPlannedSourceBlocks(blocks, &plan->source_blocks);
  NormalizeBlocks(blocks);
  auto mapped =
      MaterializeDestinationPlacement(placement, plan->handles, pool_type);
  InstallMappedDestinationRange(
      blocks, placement, std::move(mapped.free_block), pool_type);
  NormalizeBlocks(blocks);
  Commit();
  return true;
}

bool RemapTransaction::MovePlannedPagesToTargets(
    BlockList* blocks,
    SourceMovePlan* plan,
    const std::vector<VMMBackingMap::UnmappedPage>& target_pages,
    MoveCommitStats* move_stats) {
  const size_t handle_count = plan->handles.size();
  if (target_pages.size() != handle_count) {
    VLOG(3) << "VMM V2 remap transaction: MovePage target count mismatch, "
            << "targets=" << target_pages.size() << " handles=" << handle_count;
    return false;
  }
  rollback_source_mappings_ =
      [this, blocks, handles = plan->handles, metas = plan->metas]() {
        RestoreRemappedSourcesToFreeBlocks(blocks, handles, metas);
      };

  VMMDevicePtr source_range_start = 0;
  VMMDevicePtr source_expected_next = 0;
  size_t source_range_first = 0;
  size_t source_range_handles = 0;
  auto flush_source_unmap_range = [&]() {
    if (source_range_handles == 0) {
      return true;
    }
    if (move_stats != nullptr) {
      move_stats->unmap_ranges += 1;
    }
    const size_t max_chunk_handles = RemapSourceUnmapChunkHandles(handle_size_);
    size_t chunk_offset = 0;
    while (chunk_offset < source_range_handles) {
      const size_t chunk_handles =
          std::min(max_chunk_handles, source_range_handles - chunk_offset);
      const VMMDevicePtr chunk_start =
          source_range_start + chunk_offset * handle_size_;
      CUDAVirtualMemAllocatorV2::MoveBackingPageStats unmap_stats;
      const bool ok = vmm_allocator_->UnmapMappedRangeForRemap(
          chunk_start, chunk_handles, &unmap_stats);
      if (move_stats != nullptr) {
        move_stats->unmap_us += unmap_stats.unmap_us;
        move_stats->metadata_us += unmap_stats.metadata_us;
        move_stats->unmap_calls += unmap_stats.unmap_calls;
      }
      if (!ok) {
        VLOG(3) << "VMM V2 remap transaction: batched source unmap failed "
                << "range_start=" << reinterpret_cast<void*>(source_range_start)
                << " chunk_start=" << reinterpret_cast<void*>(chunk_start)
                << " chunk_handles=" << chunk_handles
                << " range_handles=" << source_range_handles;
        Rollback();
        return false;
      }
      auto metadata_start = Clock::now();
      for (size_t i = source_range_first + chunk_offset;
           i < source_range_first + chunk_offset + chunk_handles;
           ++i) {
        plan->metas[i]->MarkOwnedByRemapDestination();
      }
      if (move_stats != nullptr) {
        move_stats->metadata_us += ElapsedMicros(metadata_start, Clock::now());
      }
      chunk_offset += chunk_handles;
    }
    source_range_start = 0;
    source_expected_next = 0;
    source_range_first = 0;
    source_range_handles = 0;
    return true;
  };
  for (size_t i = 0; i < handle_count; ++i) {
    const auto& source_page = plan->source_pages[i];
    if (source_range_handles == 0) {
      source_range_start = source_page.va;
      source_expected_next = source_page.va + handle_size_;
      source_range_first = i;
      source_range_handles = 1;
      continue;
    }
    if (source_page.va == source_expected_next) {
      source_expected_next += handle_size_;
      ++source_range_handles;
      continue;
    }
    if (!flush_source_unmap_range()) {
      return false;
    }
    source_range_start = source_page.va;
    source_expected_next = source_page.va + handle_size_;
    source_range_first = i;
    source_range_handles = 1;
  }
  if (!flush_source_unmap_range()) {
    return false;
  }

  for (size_t i = 0; i < handle_count; ++i) {
    CUDAVirtualMemAllocatorV2::MoveBackingPageStats page_stats;
    if (!vmm_allocator_->MoveBackingPageForRemap(plan->source_pages[i],
                                                 target_pages[i],
                                                 plan->metas[i],
                                                 &page_stats,
                                                 true,
                                                 true)) {
      VLOG(3) << "VMM V2 remap transaction: MoveBackingPage failed at " << i
              << "/" << handle_count
              << " source=" << reinterpret_cast<void*>(plan->source_pages[i].va)
              << " target=" << reinterpret_cast<void*>(target_pages[i].va)
              << " handle="
              << reinterpret_cast<void*>(plan->source_pages[i].handle)
              << " meta=" << plan->metas[i].get();
      Rollback();
      return false;
    }
    if (move_stats != nullptr) {
      move_stats->unmap_us += page_stats.unmap_us;
      move_stats->map_us += page_stats.map_us;
      move_stats->set_access_us += page_stats.set_access_us;
      move_stats->metadata_us += page_stats.metadata_us;
      move_stats->restore_us += page_stats.restore_us;
      move_stats->rollback_us += page_stats.rollback_us;
      move_stats->unmap_calls += page_stats.unmap_calls;
      move_stats->set_access_calls += page_stats.set_access_calls;
    }
    RecordMappedDestinationRange(target_pages[i].va, 1);
  }
  VMMDevicePtr range_start = 0;
  VMMDevicePtr expected_next = 0;
  size_t range_handles = 0;
  auto flush_target_access_range = [&]() {
    if (range_handles == 0) {
      return true;
    }
    CUDAVirtualMemAllocatorV2::MoveBackingPageStats access_stats;
    const size_t range_size = range_handles * handle_size_;
    const bool ok = vmm_allocator_->SetAccessForMappedRange(
        range_start, range_size, &access_stats);
    if (move_stats != nullptr) {
      move_stats->set_access_ranges += 1;
      move_stats->set_access_us += access_stats.set_access_us;
      move_stats->set_access_calls += access_stats.set_access_calls;
    }
    if (!ok) {
      VLOG(3) << "VMM V2 remap transaction: batched target SetAccess failed "
              << "range_start=" << reinterpret_cast<void*>(range_start)
              << " handles=" << range_handles;
      Rollback();
      return false;
    }
    range_start = 0;
    expected_next = 0;
    range_handles = 0;
    return true;
  };
  for (const auto& target_page : target_pages) {
    if (range_handles == 0) {
      range_start = target_page.va;
      expected_next = target_page.va + handle_size_;
      range_handles = 1;
      continue;
    }
    if (target_page.va == expected_next) {
      expected_next += handle_size_;
      ++range_handles;
      continue;
    }
    if (!flush_target_access_range()) {
      return false;
    }
    range_start = target_page.va;
    expected_next = target_page.va + handle_size_;
    range_handles = 1;
  }
  if (!flush_target_access_range()) {
    return false;
  }
  return true;
}

bool RemapTransaction::TryCommitSingleUnmappedFreeMovePlacement(
    BlockList* blocks,
    BlockIterator unmapped_free_it,
    SourceMovePlan* plan,
    PoolType pool_type,
    MoveCommitStats* move_stats) {
  const size_t handle_count = plan->handles.size();
  const VMMDevicePtr unmapped_free_va = unmapped_free_it->begin_va();
  auto placement = DestinationPlacement::UnmappedFree(
      unmapped_free_it, unmapped_free_va, 0, handle_count);
  std::vector<VMMBackingMap::UnmappedPage> target_pages;
  if (!PrepareMoveDestinationPlacement(
          placement,
          "RemapTransaction::TryCommitSingleUnmappedFreeMovePlacement",
          &target_pages)) {
    return false;
  }

  if (!MovePlannedPagesToTargets(blocks, plan, target_pages, move_stats)) {
    return false;
  }
  ApplyPlannedSourceBlocks(blocks, &plan->source_blocks);
  auto mapped =
      MaterializeDestinationPlacement(placement, plan->handles, pool_type);
  InstallMappedDestinationRange(
      blocks, placement, std::move(mapped.free_block), pool_type);
  NormalizeBlocks(blocks);
  Commit();
  return true;
}

bool RemapTransaction::TryCommitUnmappedFreeMoveScatter(
    BlockList* blocks,
    SourceMovePlan* plan,
    const std::vector<DestinationPlacement>& placements,
    PoolType pool_type,
    MoveCommitStats* move_stats) {
  std::vector<VMMBackingMap::UnmappedPage> target_pages;
  target_pages.reserve(plan->handles.size());
  for (const auto& p : placements) {
    std::vector<VMMBackingMap::UnmappedPage> pages;
    if (!PrepareMoveDestinationPlacement(
            p, "RemapTransaction::TryCommitUnmappedFreeMoveScatter", &pages)) {
      return false;
    }
    target_pages.insert(target_pages.end(), pages.begin(), pages.end());
  }

  if (!MovePlannedPagesToTargets(blocks, plan, target_pages, move_stats)) {
    return false;
  }
  ApplyPlannedSourceBlocks(blocks, &plan->source_blocks);
  for (const auto& p : placements) {
    auto mapped = MaterializeDestinationPlacement(p, plan->handles, pool_type);
    InstallMappedDestinationRange(
        blocks, p, std::move(mapped.free_block), pool_type);
  }
  NormalizeBlocks(blocks);
  Commit();
  return true;
}

RemapTransaction::PlacementResult
RemapTransaction::ExecuteMovePlacementStrategy(BlockList* blocks,
                                               VMMDevicePtr tail_va,
                                               VMMDevicePtr va_limit,
                                               SourceMovePlan* plan,
                                               PoolType pool_type) {
  PlacementResult result;
  auto plan_start = Clock::now();
  auto destination = SelectDestinationPlan(
      blocks, tail_va, va_limit, plan->handles.size(), "MovePage");
  result.destination_plan_us = ElapsedMicros(plan_start, Clock::now());
  if (!destination.HasPlacement()) {
    return result;
  }
  result.target_min_va = std::numeric_limits<VMMDevicePtr>::max();
  result.target_max_va = 0;
  for (const auto& placement : destination.placements) {
    if (placement.count == 0) {
      continue;
    }
    result.target_min_va = std::min(result.target_min_va, placement.dst);
    result.target_max_va = std::max(
        result.target_max_va, placement.dst + placement.count * handle_size_);
  }

  auto commit_start = Clock::now();
  switch (destination.kind) {
    case DestinationPlanKind::kTail:
      result.success =
          TryCommitTailMovePlacement(blocks,
                                     destination.placements.front().dst,
                                     plan,
                                     pool_type,
                                     &result.move_stats);
      result.used_tail = result.success;
      result.move_commit_us = ElapsedMicros(commit_start, Clock::now());
      return result;
    case DestinationPlanKind::kSingleUnmappedFree:
      result.success = TryCommitSingleUnmappedFreeMovePlacement(
          blocks,
          destination.placements.front().unmapped_free_it,
          plan,
          pool_type,
          &result.move_stats);
      result.move_commit_us = ElapsedMicros(commit_start, Clock::now());
      return result;
    case DestinationPlanKind::kScatterUnmappedFree:
      result.success = TryCommitUnmappedFreeMoveScatter(
          blocks, plan, destination.placements, pool_type, &result.move_stats);
      result.move_commit_us = ElapsedMicros(commit_start, Clock::now());
      return result;
    case DestinationPlanKind::kNone:
      return result;
  }
  return result;
}

RemapTransaction::CompactResult RemapTransaction::CompactFreeBlocks(
    BlockList* blocks, size_t requested_size, PoolType pool_type) {
  CompactResult result;
  rollback_source_mappings_ = {};

  VMMDevicePtr tail_va = vmm_allocator_->virtual_mem_base();
  if (!blocks->empty()) {
    const auto& last = blocks->back();
    tail_va = last.end_va();
  }
  const VMMDevicePtr va_limit =
      vmm_allocator_->virtual_mem_base() + vmm_allocator_->virtual_mem_size();

  auto source_start = Clock::now();
  auto move_plan = CollectRemapSourcePlan(blocks, requested_size, pool_type);
  result.source_collect_us = ElapsedMicros(source_start, Clock::now());
  result.source_stats = move_plan.stats;
  result.remapped_handle_count = move_plan.handles.size();
  result.remapped_bytes = move_plan.handles.size() * handle_size_;
  if (!move_plan.source_pages.empty()) {
    result.source_min_va = std::numeric_limits<VMMDevicePtr>::max();
    result.source_max_va = 0;
    for (const auto& page : move_plan.source_pages) {
      result.source_min_va = std::min(result.source_min_va, page.va);
      result.source_max_va =
          std::max(result.source_max_va, page.va + handle_size_);
    }
  }
  if (move_plan.handles.empty()) {
    return result;
  }
  auto move_placement = ExecuteMovePlacementStrategy(
      blocks, tail_va, va_limit, &move_plan, pool_type);
  result.success = move_placement.success;
  result.used_tail = move_placement.used_tail;
  result.destination_plan_us = move_placement.destination_plan_us;
  result.move_commit_us = move_placement.move_commit_us;
  result.move_stats = move_placement.move_stats;
  result.target_min_va = move_placement.target_min_va;
  result.target_max_va = move_placement.target_max_va;
  if (result.success) {
    if (result.used_tail) {
      vmm_allocator_->AdvanceTailOffset(result.remapped_bytes);
    }
  }
  return result;
}

void RemapTransaction::InstallTailFreeBlock(BlockList* blocks,
                                            BlockV2 free_block) const {
  if (!blocks->empty()) {
    auto last = std::prev(blocks->end());
    if (last->CanMergeAdjacentFreeBlock(free_block)) {
      last->MergeAdjacentBlock(free_block);
      return;
    }
  }
  blocks->push_back(std::move(free_block));
}

RemapTransaction::BlockIterator
RemapTransaction::InstallMappedUnmappedFreeRange(BlockList* blocks,
                                                 BlockIterator unmapped_free_it,
                                                 BlockV2 free_block,
                                                 PoolType pool_type) const {
  VMMDevicePtr unmapped_free_va = unmapped_free_it->begin_va();
  size_t unmapped_free_size = unmapped_free_it->size();
  size_t filled_bytes = free_block.size();

  if (unmapped_free_size == filled_bytes) {
    *unmapped_free_it = std::move(free_block);
  } else {
    BlockV2 remaining_unmapped_free = MakeUnmappedFreeBlock(
        reinterpret_cast<void*>(unmapped_free_va + filled_bytes),
        unmapped_free_size - filled_bytes,
        pool_type);
    *unmapped_free_it = std::move(free_block);
    blocks->insert(std::next(unmapped_free_it),
                   std::move(remaining_unmapped_free));
  }

  auto result = unmapped_free_it;
  if (result != blocks->begin()) {
    auto prev = std::prev(result);
    if (prev->CanMergeAdjacentFreeBlock(*result)) {
      prev->MergeAdjacentBlock(*result);
      blocks->erase(result);
      result = prev;
    }
  }

  auto next = std::next(result);
  if (next != blocks->end() && result->CanMergeAdjacentFreeBlock(*next)) {
    result->MergeAdjacentBlock(*next);
    blocks->erase(next);
  }
  return result;
}

void RemapTransaction::InstallMappedDestinationRange(
    BlockList* blocks,
    const DestinationPlacement& placement,
    BlockV2 free_block,
    PoolType pool_type) const {
  if (placement.is_tail) {
    InstallTailFreeBlock(blocks, std::move(free_block));
    return;
  }
  InstallMappedUnmappedFreeRange(
      blocks, placement.unmapped_free_it, std::move(free_block), pool_type);
}

BlockV2 RemapTransaction::MakeUnmappedFreeBlock(void* ptr,
                                                size_t size,
                                                PoolType pool_type) const {
  return BlockV2::MakeUnmappedFreeBlock(ptr, size, pool_type);
}

void RemapTransaction::MergeAdjacentFreeBlocks(BlockList* blocks) const {
  for (auto it = blocks->begin(); it != blocks->end();) {
    if (!it->IsFree()) {
      ++it;
      continue;
    }
    auto next = std::next(it);
    if (next != blocks->end() && it->CanMergeAdjacentFreeBlock(*next)) {
      it->MergeAdjacentBlock(*next);
      blocks->erase(next);
      continue;
    }
    ++it;
  }
}

void RemapTransaction::MergeAdjacentUnmappedFreeBlocks(
    BlockList* blocks) const {
  for (auto it = blocks->begin(); it != blocks->end();) {
    auto next = std::next(it);
    if (next != blocks->end() && it->CanMergeAdjacentUnmappedFreeBlock(*next)) {
      it->MergeAdjacentUnmappedFreeBlock(*next);
      blocks->erase(next);
      continue;
    }
    ++it;
  }
}

void RemapTransaction::NormalizeBlocks(BlockList* blocks) const {
  MergeAdjacentFreeBlocks(blocks);
  MergeAdjacentUnmappedFreeBlocks(blocks);
}

bool RemapTransaction::RestoreUnmappedFreeRangeToMappedFreeBlock(
    BlockList* blocks,
    VMMDevicePtr va,
    size_t size,
    const std::shared_ptr<VMMHandleMeta>& meta) {
  for (auto& block : *blocks) {
    if (!block.IsFree()) {
      continue;
    }
    if (block.ContainsVARange(va, size)) {
      return true;
    }
  }
  for (auto it = blocks->begin(); it != blocks->end(); ++it) {
    std::vector<BlockV2> replacement_segments;
    auto restore_result = it->BuildRestoreMappedFreeSegments(
        va, size, meta, &replacement_segments);
    if (restore_result == BlockRestoreMappedFreeResult::kOutside) {
      continue;
    }
    if (restore_result == BlockRestoreMappedFreeResult::kRangeExceedsBlock) {
      VLOG(3) << "RestoreUnmappedFreeRangeToMappedFreeBlock: range exceeds "
                 "unmapped-free block, va="
              << reinterpret_cast<void*>(va) << " size=" << size
              << " block_start=" << reinterpret_cast<void*>(it->begin_va())
              << " unmapped_free_size=" << it->size();
      return false;
    }

    auto insert_pos = it;
    for (auto& segment : replacement_segments) {
      blocks->insert(insert_pos, std::move(segment));
    }
    blocks->erase(it);
    return true;
  }
  VLOG(3) << "RestoreUnmappedFreeRangeToMappedFreeBlock: unmapped-free block "
             "not found for VA "
          << reinterpret_cast<void*>(va) << "; force-release will follow";
  return false;
}

bool RemapTransaction::RestoreRemappedSourcesToFreeBlocks(
    BlockList* blocks,
    const std::vector<VMMAllocHandle>& handles,
    const std::vector<std::shared_ptr<VMMHandleMeta>>& metas) {
  if (handles.empty()) {
    return false;
  }
  RestoreSourceMappings(
      blocks,
      handles,
      metas,
      vmm_allocator_,
      handle_size_,
      [this](BlockList* restore_blocks) {
        MergeAdjacentFreeBlocks(restore_blocks);
      },
      [this](BlockList* restore_blocks,
             VMMDevicePtr va,
             size_t size,
             const std::shared_ptr<VMMHandleMeta>& meta) {
        return RestoreUnmappedFreeRangeToMappedFreeBlock(
            restore_blocks, va, size, meta);
      });
  return true;
}

void RemapTransaction::RecordMappedDestinationRange(VMMDevicePtr dst,
                                                    size_t handle_count) {
  rollback_mapped_destinations_.push_back([this, dst, handle_count]() {
    VLOG(3) << "VMM V2 remap transaction: unmapping mapped destination "
            << reinterpret_cast<void*>(dst) << " handles=" << handle_count;
    vmm_allocator_->RollbackMappedHandleRange(dst, handle_count);
  });
}

void RemapTransaction::Commit() {
  PADDLE_ENFORCE_EQ(
      pending_synthetic_allocations_.empty() ||
          static_cast<bool>(commit_synthetic_allocation_),
      true,
      common::errors::InvalidArgument(
          "RemapTransaction committed %d staged synthetic allocations without "
          "an ownership sink.",
          pending_synthetic_allocations_.size()));
  if (commit_synthetic_allocation_) {
    while (!pending_synthetic_allocations_.empty()) {
      auto* allocation = pending_synthetic_allocations_.back();
      pending_synthetic_allocations_.pop_back();
      try {
        auto committed_allocation =
            vmm_allocator_->AdoptCommittedSyntheticAllocation(allocation);
        allocation = nullptr;
        commit_synthetic_allocation_(std::move(committed_allocation));
      } catch (...) {
        if (allocation != nullptr) {
          vmm_allocator_->DestroyStagedSyntheticAllocation(allocation);
        }
        throw;
      }
    }
  }
  rollback_mapped_destinations_.clear();
  rollback_source_mappings_ = {};
  completed_ = true;
}

void RemapTransaction::Rollback() {
  if (completed_) {
    return;
  }
  while (!pending_synthetic_allocations_.empty()) {
    auto* allocation = pending_synthetic_allocations_.back();
    pending_synthetic_allocations_.pop_back();
    vmm_allocator_->DestroyStagedSyntheticAllocation(allocation);
  }
  RollbackMappedDestinations();
  if (rollback_source_mappings_) {
    rollback_source_mappings_();
  }
  rollback_source_mappings_ = {};
  completed_ = true;
}

void RemapTransaction::RollbackMappedDestinations() {
  for (auto it = rollback_mapped_destinations_.rbegin();
       it != rollback_mapped_destinations_.rend();
       ++it) {
    (*it)();
  }
  rollback_mapped_destinations_.clear();
}

void RemapTransaction::StageSyntheticAllocation(Allocation* allocation) {
  pending_synthetic_allocations_.emplace_back(allocation);
}

bool RemapTransaction::HasPendingState() const {
  return !pending_synthetic_allocations_.empty() ||
         !rollback_mapped_destinations_.empty() ||
         static_cast<bool>(rollback_source_mappings_);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
