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

#if defined(PADDLE_WITH_CUDA)

#include <algorithm>
#include <chrono>
#include <limits>
#include <list>
#include <memory>
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

SourcePageMap IndexSourcePageCandidates(
    const RemapTransaction::SourcePages& source_pages) {
  SourcePageMap candidates;
  candidates.reserve(source_pages.size());
  for (const auto& page : source_pages) {
    candidates.emplace(page.va, page);
  }
  return candidates;
}

}  // namespace

RemapTransaction::~RemapTransaction() {
  if (!completed_ && HasPendingState()) {
    VLOG(3) << "VMM V2 remap transaction destroyed before Commit/Rollback; "
               "rolling back pending state";
    Rollback();
  }
}

bool RemapTransaction::CheckBlockRemapSafety(BlockV2* block) {
  if (!block->IsMappedFree() || block->HasUnknownRemapSafety()) {
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

BlockV2 RemapTransaction::MaterializeDestinationRange(
    VMMDevicePtr dst,
    const SourcePages& source_pages,
    size_t start,
    size_t count,
    PoolType pool_type) {
  CUDAVirtualMemAllocatorV2::StagedRemapDestination staged;
  staged = vmm_allocator_->CreateStagedRemapDestination(
      dst, source_pages, start, count, pool_type);
  auto destroy_staged = [this](Allocation* allocation) {
    vmm_allocator_->DestroyStagedDestinationAllocation(allocation);
  };
  std::unique_ptr<Allocation, decltype(destroy_staged)> staged_guard(
      staged.allocation, destroy_staged);
  StageDestinationAllocation(staged.allocation);
  staged_guard.release();
  return std::move(staged.block);
}

BlockV2 RemapTransaction::MaterializeDestinationPlacement(
    const DestinationPlacement& placement,
    const SourcePages& source_pages,
    PoolType pool_type) {
  return MaterializeDestinationRange(placement.dst,
                                     source_pages,
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

bool RemapTransaction::PrepareDestinationPlacement(
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

bool RemapTransaction::CanPrepareDestinationRange(VMMDevicePtr dst,
                                                  size_t size) const {
  if (!can_prepare_destination_range_) {
    return true;
  }
  return can_prepare_destination_range_(reinterpret_cast<void*>(dst), size);
}

bool RemapTransaction::PrepareDestinationRange(VMMDevicePtr dst,
                                               size_t size,
                                               const char* context) const {
  if (!prepare_destination_range_ ||
      prepare_destination_range_(reinterpret_cast<void*>(dst), size)) {
    return true;
  }
  VLOG(3) << "VMM V2 remap transaction: destination preparation "
             "failed in "
          << context << ", dst=" << reinterpret_cast<void*>(dst)
          << " bytes=" << size;
  return false;
}

RemapTransaction::SourceMovePlan RemapTransaction::CollectRemapSourcePlan(
    BlockList* blocks,
    size_t requested_size,
    PoolType pool_type,
    const SourcePages& source_pages) {
  SourceMovePlan plan;
  auto source_candidates = IndexSourcePageCandidates(source_pages);
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
    if (!CheckBlockRemapSafety(&*current)) {
      plan.stats.event_blocked_count +=
          (current->size() + handle_size_ - 1) / handle_size_;
      plan.stats.event_blocked_bytes += current->size();
      continue;
    }
    plan.stats.safe_block_count++;

    std::vector<BlockV2> replacement_segments;
    size_t remapped_count_before = plan.source_pages.size();
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
        plan.source_pages.push_back(candidate);
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

    if (plan.source_pages.size() == remapped_count_before) {
      continue;
    }
    plan.source_blocks.push_back({current, std::move(replacement_segments)});

    if (requested_size > 0 &&
        plan.source_pages.size() * handle_size_ >= requested_size) {
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

RemapTransaction::UnmappedDestinationPlan
RemapTransaction::PlanUnmappedDestinations(BlockList* blocks,
                                           size_t handle_count) const {
  UnmappedDestinationPlan plan;
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
          CanPrepareDestinationRange(it->begin_va(), required_bytes)) {
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
    if (CanPrepareDestinationRange(dst, capacity_bytes)) {
      plan.total_capacity += capacity_bytes;
    }

    if (handle_idx >= handle_count) continue;
    size_t to_fill = std::min(unmapped_free_cap, handle_count - handle_idx);
    if (!CanPrepareDestinationRange(dst, to_fill * handle_size_)) {
      continue;
    }
    plan.scatter_placements.push_back(
        DestinationPlacement::InUnmappedRange(it, dst, handle_idx, to_fill));
    handle_idx += to_fill;
  }
  plan.scatter_handles = handle_idx;
  return plan;
}

RemapTransaction::DestinationPlan RemapTransaction::SelectDestinationPlan(
    BlockList* blocks,
    VMMDevicePtr tail_va,
    VMMDevicePtr va_limit,
    size_t handle_count) const {
  DestinationPlan plan;
  const size_t total_remapped = handle_count * handle_size_;

  const bool tail_driver_va_usable =
      TailIsUsable(tail_va, total_remapped, va_limit);
  if (tail_driver_va_usable &&
      CanPrepareDestinationRange(tail_va, total_remapped)) {
    plan.kind = DestinationPlanKind::kTail;
    plan.placements.push_back(
        DestinationPlacement::AtTail(tail_va, handle_count));
    return plan;
  }

  auto unmapped_plan = PlanUnmappedDestinations(blocks, handle_count);
  if (unmapped_plan.single_it != blocks->end()) {
    BlockIterator unmapped_it = unmapped_plan.single_it;
    const VMMDevicePtr unmapped_va = unmapped_it->begin_va();
    plan.kind = DestinationPlanKind::kSingleUnmappedRange;
    plan.placements.push_back(DestinationPlacement::InUnmappedRange(
        unmapped_it, unmapped_va, 0, handle_count));
    return plan;
  }

  plan.kind = DestinationPlanKind::kScatterUnmappedRanges;
  plan.placements = std::move(unmapped_plan.scatter_placements);
  if (unmapped_plan.scatter_handles == handle_count) {
    return plan;
  }
  if (unmapped_plan.total_capacity < total_remapped) {
    plan.kind = DestinationPlanKind::kNone;
    return plan;
  }

  plan.kind = DestinationPlanKind::kNone;
  return plan;
}

bool RemapTransaction::TryMoveToTail(BlockList* blocks,
                                     VMMDevicePtr tail_va,
                                     SourceMovePlan* plan,
                                     PoolType pool_type,
                                     MoveCommitStats* move_stats) {
  const size_t handle_count = plan->source_pages.size();
  auto placement = DestinationPlacement::AtTail(tail_va, handle_count);
  std::vector<VMMBackingMap::UnmappedPage> target_pages;
  if (!PrepareDestinationPlacement(
          placement, "RemapTransaction::TryMoveToTail", &target_pages)) {
    return false;
  }

  if (!MovePlannedPagesToTargets(blocks, plan, target_pages, move_stats)) {
    return false;
  }
  ApplyPlannedSourceBlocks(blocks, &plan->source_blocks);
  NormalizeBlocks(blocks);
  auto mapped_free_block =
      MaterializeDestinationPlacement(placement, plan->source_pages, pool_type);
  InstallDestination(
      blocks, placement, std::move(mapped_free_block), pool_type);
  NormalizeBlocks(blocks);
  Commit();
  return true;
}

bool RemapTransaction::MovePlannedPagesToTargets(
    BlockList* blocks,
    SourceMovePlan* plan,
    const std::vector<VMMBackingMap::UnmappedPage>& target_pages,
    MoveCommitStats* move_stats) {
  const size_t handle_count = plan->source_pages.size();
  if (target_pages.size() != handle_count) {
    VLOG(3) << "VMM V2 remap transaction: MovePage target count mismatch, "
            << "targets=" << target_pages.size() << " handles=" << handle_count;
    return false;
  }
  rollback_source_mappings_ =
      [this, blocks, source_pages = plan->source_pages]() {
        RestoreRemappedSourcesToFreeBlocks(blocks, source_pages);
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
        Rollback();
        return false;
      }
      auto metadata_start = Clock::now();
      for (size_t i = source_range_first + chunk_offset;
           i < source_range_first + chunk_offset + chunk_handles;
           ++i) {
        plan->source_pages[i].meta->MarkOwnedByRemapDestination();
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
                                                 plan->source_pages[i].meta,
                                                 &page_stats,
                                                 true,
                                                 true)) {
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
    RecordDestinationRollbackRange(target_pages[i].va, 1);
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

bool RemapTransaction::TryMoveToUnmappedRange(BlockList* blocks,
                                              BlockIterator unmapped_it,
                                              SourceMovePlan* plan,
                                              PoolType pool_type,
                                              MoveCommitStats* move_stats) {
  const size_t handle_count = plan->source_pages.size();
  const VMMDevicePtr unmapped_va = unmapped_it->begin_va();
  auto placement = DestinationPlacement::InUnmappedRange(
      unmapped_it, unmapped_va, 0, handle_count);
  std::vector<VMMBackingMap::UnmappedPage> target_pages;
  if (!PrepareDestinationPlacement(placement,
                                   "RemapTransaction::TryMoveToUnmappedRange",
                                   &target_pages)) {
    return false;
  }

  if (!MovePlannedPagesToTargets(blocks, plan, target_pages, move_stats)) {
    return false;
  }
  ApplyPlannedSourceBlocks(blocks, &plan->source_blocks);
  auto mapped_free_block =
      MaterializeDestinationPlacement(placement, plan->source_pages, pool_type);
  InstallDestination(
      blocks, placement, std::move(mapped_free_block), pool_type);
  NormalizeBlocks(blocks);
  Commit();
  return true;
}

bool RemapTransaction::TryMoveToUnmappedRanges(
    BlockList* blocks,
    SourceMovePlan* plan,
    const std::vector<DestinationPlacement>& placements,
    PoolType pool_type,
    MoveCommitStats* move_stats) {
  std::vector<VMMBackingMap::UnmappedPage> target_pages;
  target_pages.reserve(plan->source_pages.size());
  for (const auto& p : placements) {
    std::vector<VMMBackingMap::UnmappedPage> pages;
    if (!PrepareDestinationPlacement(
            p, "RemapTransaction::TryMoveToUnmappedRanges", &pages)) {
      return false;
    }
    target_pages.insert(target_pages.end(), pages.begin(), pages.end());
  }

  if (!MovePlannedPagesToTargets(blocks, plan, target_pages, move_stats)) {
    return false;
  }
  ApplyPlannedSourceBlocks(blocks, &plan->source_blocks);
  for (const auto& p : placements) {
    auto mapped_free_block =
        MaterializeDestinationPlacement(p, plan->source_pages, pool_type);
    InstallDestination(blocks, p, std::move(mapped_free_block), pool_type);
  }
  NormalizeBlocks(blocks);
  Commit();
  return true;
}

RemapTransaction::MoveResult RemapTransaction::ExecuteDestinationPlan(
    BlockList* blocks,
    VMMDevicePtr tail_va,
    VMMDevicePtr va_limit,
    SourceMovePlan* plan,
    PoolType pool_type) {
  MoveResult result;
  auto plan_start = Clock::now();
  auto destination = SelectDestinationPlan(
      blocks, tail_va, va_limit, plan->source_pages.size());
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
      result.success = TryMoveToTail(blocks,
                                     destination.placements.front().dst,
                                     plan,
                                     pool_type,
                                     &result.move_stats);
      result.used_tail = result.success;
      result.move_commit_us = ElapsedMicros(commit_start, Clock::now());
      return result;
    case DestinationPlanKind::kSingleUnmappedRange:
      result.success =
          TryMoveToUnmappedRange(blocks,
                                 destination.placements.front().unmapped_it,
                                 plan,
                                 pool_type,
                                 &result.move_stats);
      result.move_commit_us = ElapsedMicros(commit_start, Clock::now());
      return result;
    case DestinationPlanKind::kScatterUnmappedRanges:
      result.success = TryMoveToUnmappedRanges(
          blocks, plan, destination.placements, pool_type, &result.move_stats);
      result.move_commit_us = ElapsedMicros(commit_start, Clock::now());
      return result;
    case DestinationPlanKind::kNone:
      return result;
  }
  return result;
}

RemapTransaction::CompactResult RemapTransaction::CompactFreeBlocks(
    BlockList* blocks,
    size_t requested_size,
    PoolType pool_type,
    const SourcePages& source_pages) {
  CompactResult result;
  blocks_ = blocks;
  rollback_source_mappings_ = {};

  VMMDevicePtr tail_va = vmm_allocator_->virtual_mem_base();
  if (!blocks->empty()) {
    const auto& last = blocks->back();
    tail_va = last.end_va();
  }
  const VMMDevicePtr va_limit =
      vmm_allocator_->virtual_mem_base() + vmm_allocator_->virtual_mem_size();

  auto source_start = Clock::now();
  auto move_plan =
      CollectRemapSourcePlan(blocks, requested_size, pool_type, source_pages);
  result.source_collect_us = ElapsedMicros(source_start, Clock::now());
  result.source_stats = move_plan.stats;
  result.remapped_handle_count = move_plan.source_pages.size();
  result.remapped_bytes = move_plan.source_pages.size() * handle_size_;
  if (!move_plan.source_pages.empty()) {
    result.source_min_va = std::numeric_limits<VMMDevicePtr>::max();
    result.source_max_va = 0;
    for (const auto& page : move_plan.source_pages) {
      result.source_min_va = std::min(result.source_min_va, page.va);
      result.source_max_va =
          std::max(result.source_max_va, page.va + handle_size_);
    }
  }
  if (move_plan.source_pages.empty()) {
    return result;
  }
  auto move_placement =
      ExecuteDestinationPlan(blocks, tail_va, va_limit, &move_plan, pool_type);
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
RemapTransaction::ReplaceUnmappedRangeWithMappedFree(BlockList* blocks,
                                                     BlockIterator unmapped_it,
                                                     BlockV2 mapped_free_block,
                                                     PoolType pool_type) const {
  VMMDevicePtr unmapped_va = unmapped_it->begin_va();
  size_t unmapped_size = unmapped_it->size();
  size_t filled_bytes = mapped_free_block.size();

  if (unmapped_size == filled_bytes) {
    *unmapped_it = std::move(mapped_free_block);
  } else {
    BlockV2 remaining_unmapped_free = BlockV2::MakeUnmappedFreeBlock(
        reinterpret_cast<void*>(unmapped_va + filled_bytes),
        unmapped_size - filled_bytes,
        pool_type);
    *unmapped_it = std::move(mapped_free_block);
    blocks->insert(std::next(unmapped_it), std::move(remaining_unmapped_free));
  }

  auto result = unmapped_it;
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

void RemapTransaction::InstallDestination(BlockList* blocks,
                                          const DestinationPlacement& placement,
                                          BlockV2 mapped_free_block,
                                          PoolType pool_type) {
  RecordDestinationBlockRollbackRange(placement);
  if (placement.is_tail) {
    InstallTailFreeBlock(blocks, std::move(mapped_free_block));
    return;
  }
  ReplaceUnmappedRangeWithMappedFree(
      blocks, placement.unmapped_it, std::move(mapped_free_block), pool_type);
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

bool RemapTransaction::RestoreRangeAsMappedFree(BlockList* blocks,
                                                VMMDevicePtr va,
                                                size_t size) {
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
    auto restore_result =
        it->BuildRestoreMappedFreeSegments(va, size, &replacement_segments);
    if (restore_result == BlockRestoreMappedFreeResult::kOutside) {
      continue;
    }
    if (restore_result == BlockRestoreMappedFreeResult::kRangeExceedsBlock) {
      VLOG(3) << "RestoreRangeAsMappedFree: range exceeds "
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
  VLOG(3) << "RestoreRangeAsMappedFree: unmapped-free block "
             "not found for VA "
          << reinterpret_cast<void*>(va) << "; force-release will follow";
  return false;
}

bool RemapTransaction::ReplaceRangeWithUnmappedFree(BlockList* blocks,
                                                    VMMDevicePtr va,
                                                    size_t size) {
  for (auto it = blocks->begin(); it != blocks->end(); ++it) {
    if (!it->ContainsVARange(va, size)) {
      continue;
    }
    if (it->IsUnmappedFree()) {
      return true;
    }
    if (!it->IsMappedFree()) {
      return false;
    }

    const VMMDevicePtr block_begin = it->begin_va();
    const VMMDevicePtr block_end = it->end_va();
    std::vector<BlockV2> replacement_segments;
    AppendMappedFreeSubRange(
        &replacement_segments, *it, block_begin, va - block_begin);
    AppendUnmappedFreeRange(&replacement_segments, va, size, it->pool_type_);
    AppendMappedFreeSubRange(
        &replacement_segments, *it, va + size, block_end - (va + size));

    auto insert_pos = it;
    for (auto& segment : replacement_segments) {
      blocks->insert(insert_pos, std::move(segment));
    }
    blocks->erase(it);
    return true;
  }
  return false;
}

bool RemapTransaction::RemoveTailDestinationBlock(BlockList* blocks,
                                                  VMMDevicePtr va,
                                                  size_t size) {
  const VMMDevicePtr end = va + size;
  for (auto it = blocks->begin(); it != blocks->end(); ++it) {
    if (it->end_va() <= va || it->begin_va() >= end) {
      continue;
    }
    if (!it->IsMappedFree() || !it->ContainsVARange(va, size) ||
        it->end_va() != end) {
      return false;
    }
    if (it->begin_va() == va) {
      blocks->erase(it);
    } else {
      it->TrimToPrefix(va - it->begin_va());
    }
    return true;
  }
  // Installation may have failed before the tail block view was added.
  return true;
}

void RemapTransaction::RestoreRemappedSourcesToFreeBlocks(
    BlockList* blocks, const SourcePages& source_pages) {
  if (source_pages.empty()) {
    return;
  }
  size_t restored = 0;
  size_t force_released = 0;
  for (const auto& source_page : source_pages) {
    auto restore_result = vmm_allocator_->RestoreRemapSourceMapping(
        source_page.handle, source_page.meta, handle_size_);
    if (restore_result ==
        CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult::kSkipped) {
      continue;
    }
    if (restore_result ==
        CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult::kForceReleased) {
      PADDLE_ENFORCE_EQ(
          ReplaceRangeWithUnmappedFree(
              blocks, source_page.meta->base(), handle_size_),
          true,
          common::errors::PreconditionNotMet(
              "A VMM remap source at %p was force-released after its original "
              "mapping could not be restored, but the corresponding %zu-byte "
              "free block range could not be converted to unmapped-free.",
              reinterpret_cast<void*>(source_page.meta->base()),
              handle_size_));
      ++force_released;
      continue;
    }

    const VMMDevicePtr original_va = source_page.meta->base();
    if (RestoreRangeAsMappedFree(blocks, original_va, handle_size_)) {
      source_page.meta->RestoreOriginalOwnership();
      ++restored;
      continue;
    }
    vmm_allocator_->ForceReleaseRemapSource(source_page.handle,
                                            source_page.meta,
                                            handle_size_,
                                            "block restore",
                                            true);
    ++force_released;
  }
  MergeAdjacentFreeBlocks(blocks);
  VLOG(4) << "RestoreRemappedSourcesToFreeBlocks: restored=" << restored
          << " force_released=" << force_released;
}

void RemapTransaction::RecordDestinationRollbackRange(VMMDevicePtr dst,
                                                      size_t handle_count) {
  destination_rollback_ranges_.push_back({dst, handle_count});
}

void RemapTransaction::RecordDestinationBlockRollbackRange(
    const DestinationPlacement& placement) {
  destination_block_rollback_ranges_.push_back(
      {placement.dst, placement.count * handle_size_, placement.is_tail});
}

void RemapTransaction::DiscardDestinationRollbackRange(VMMDevicePtr dst,
                                                       size_t size) {
  const VMMDevicePtr end = dst + size;
  destination_rollback_ranges_.erase(
      std::remove_if(destination_rollback_ranges_.begin(),
                     destination_rollback_ranges_.end(),
                     [&](const DestinationRollbackRange& range) {
                       const VMMDevicePtr range_end =
                           range.va + range.handle_count * handle_size_;
                       return range.va >= dst && range_end <= end;
                     }),
      destination_rollback_ranges_.end());
  destination_block_rollback_ranges_.erase(
      std::remove_if(destination_block_rollback_ranges_.begin(),
                     destination_block_rollback_ranges_.end(),
                     [&](const DestinationBlockRollbackRange& range) {
                       return range.va >= dst && range.va + range.size <= end;
                     }),
      destination_block_rollback_ranges_.end());
}

void RemapTransaction::Commit() {
  PADDLE_ENFORCE_EQ(
      pending_destination_allocations_.empty() ||
          static_cast<bool>(commit_destination_allocations_),
      true,
      common::errors::InvalidArgument("RemapTransaction committed %d staged "
                                      "destination allocations without "
                                      "an ownership sink.",
                                      pending_destination_allocations_.size()));
  if (commit_destination_allocations_) {
    struct CommitRecord {
      VMMDevicePtr va;
      size_t size;
      std::vector<VMMBackingMap::MappedPage> destination_pages;
    };
    std::vector<CommitRecord> records;
    std::vector<DecoratedAllocationPtr> committed_allocations;
    records.reserve(pending_destination_allocations_.size());
    committed_allocations.reserve(pending_destination_allocations_.size());

    auto restore_pending_ownership = [&]() {
      for (auto& committed : committed_allocations) {
        if (committed != nullptr) {
          pending_destination_allocations_.push_back(committed.release());
        }
      }
    };

    while (!pending_destination_allocations_.empty()) {
      auto* allocation = pending_destination_allocations_.back();
      pending_destination_allocations_.pop_back();
      try {
        const auto allocation_va =
            reinterpret_cast<VMMDevicePtr>(allocation->ptr());
        const size_t allocation_size = allocation->size();
        auto destination_pages = vmm_allocator_->CollectMappedPages(
            {{allocation_va, allocation_size}}, allocation_size);
        PADDLE_ENFORCE_EQ(
            destination_pages.size(),
            allocation_size / handle_size_,
            common::errors::PreconditionNotMet(
                "Committed VMM remap destination at %p has %zu mapped pages, "
                "but %zu pages are required.",
                allocation->ptr(),
                destination_pages.size(),
                allocation_size / handle_size_));
        for (const auto& page : destination_pages) {
          PADDLE_ENFORCE_NOT_NULL(
              page.meta,
              common::errors::PreconditionNotMet(
                  "Committed VMM remap destination page at %p has no meta.",
                  reinterpret_cast<void*>(page.va)));
        }
        committed_allocations.push_back(
            vmm_allocator_->AdoptRemapDestinationAllocation(allocation));
        allocation = nullptr;
        records.push_back(
            {allocation_va, allocation_size, std::move(destination_pages)});
      } catch (...) {
        if (allocation != nullptr) {
          pending_destination_allocations_.push_back(allocation);
        }
        restore_pending_ownership();
        throw;
      }
    }

    try {
      commit_destination_allocations_(&committed_allocations);
      for (const auto& allocation : committed_allocations) {
        PADDLE_ENFORCE_EQ(
            allocation.get(),
            nullptr,
            common::errors::PreconditionNotMet(
                "VMM remap destination ownership sink did not consume all "
                "allocations."));
      }
    } catch (...) {
      restore_pending_ownership();
      throw;
    }

    for (const auto& record : records) {
      // The registry now owns destination cleanup. The transaction must no
      // longer unmap the same VA if a later bookkeeping step fails.
      DiscardDestinationRollbackRange(record.va, record.size);
      for (const auto& page : record.destination_pages) {
        page.meta->RestoreOriginalOwnership();
      }
    }
  }
  destination_rollback_ranges_.clear();
  destination_block_rollback_ranges_.clear();
  rollback_source_mappings_ = {};
  blocks_ = nullptr;
  completed_ = true;
}

void RemapTransaction::Rollback() {
  if (completed_) {
    return;
  }
  while (!pending_destination_allocations_.empty()) {
    auto* allocation = pending_destination_allocations_.back();
    pending_destination_allocations_.pop_back();
    vmm_allocator_->DestroyStagedDestinationAllocation(allocation);
  }
  RollbackDestinations();
  RollbackDestinationBlockViews();
  if (rollback_source_mappings_) {
    rollback_source_mappings_();
  }
  if (blocks_ != nullptr) {
    NormalizeBlocks(blocks_);
  }
  rollback_source_mappings_ = {};
  blocks_ = nullptr;
  completed_ = true;
}

void RemapTransaction::RollbackDestinations() {
  for (auto it = destination_rollback_ranges_.rbegin();
       it != destination_rollback_ranges_.rend();
       ++it) {
    vmm_allocator_->RollbackMappedHandleRange(it->va, it->handle_count);
  }
  destination_rollback_ranges_.clear();
}

void RemapTransaction::RollbackDestinationBlockViews() {
  if (destination_block_rollback_ranges_.empty()) {
    return;
  }
  PADDLE_ENFORCE_NOT_NULL(
      blocks_,
      common::errors::PreconditionNotMet(
          "A VMM remap transaction has destination block views to roll back, "
          "but its block list is unavailable."));
  for (auto it = destination_block_rollback_ranges_.rbegin();
       it != destination_block_rollback_ranges_.rend();
       ++it) {
    const bool restored =
        it->is_tail ? RemoveTailDestinationBlock(blocks_, it->va, it->size)
                    : ReplaceRangeWithUnmappedFree(blocks_, it->va, it->size);
    PADDLE_ENFORCE_EQ(
        restored,
        true,
        common::errors::PreconditionNotMet(
            "The VMM remap destination block view at %p with size %zu could "
            "not be restored during transaction rollback.",
            reinterpret_cast<void*>(it->va),
            it->size));
  }
  destination_block_rollback_ranges_.clear();
}

void RemapTransaction::StageDestinationAllocation(Allocation* allocation) {
  pending_destination_allocations_.emplace_back(allocation);
}

bool RemapTransaction::HasPendingState() const {
  return !pending_destination_allocations_.empty() ||
         !destination_rollback_ranges_.empty() ||
         !destination_block_rollback_ranges_.empty() ||
         static_cast<bool>(rollback_source_mappings_);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
