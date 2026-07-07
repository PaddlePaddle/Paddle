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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <limits>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/free_block_remap_compactor.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

using Clock = std::chrono::steady_clock;

std::atomic<uint64_t> g_vmm_v2_compact_seq{0};

uint64_t ElapsedMicros(Clock::time_point start, Clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());
}

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

struct DriverMemoryStats {
  size_t driver_actual_avail{0};
  size_t driver_actual_total{0};
  phi::gpuError_t mem_info_status{phi::gpuSuccess};
};

DriverMemoryStats CollectDriverMemoryStats(int device) {
  DriverMemoryStats stats;
  platform::CUDADeviceGuard guard(device);
  stats.mem_info_status =
      cudaMemGetInfo(&stats.driver_actual_avail, &stats.driver_actual_total);
  if (stats.mem_info_status != phi::gpuSuccess) {
    stats.driver_actual_avail = 0;
    stats.driver_actual_total = 0;
    (void)platform::GpuGetLastError();
  }
  return stats;
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

bool VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::
    AllOverlapsSatisfy(void* ptr,
                       size_t size,
                       const OverlapPredicate& predicate) const {
  for (const auto& allocation : allocations_) {
    if (!RangesOverlap(ptr, size, allocation->ptr(), allocation->size())) {
      continue;
    }
    if (!predicate(allocation)) {
      return false;
    }
  }
  return true;
}

bool VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::
    EraseOverlapsIf(void* ptr, size_t size, const OverlapPredicate& predicate) {
  bool ok = true;
  for (auto it = allocations_.begin(); it != allocations_.end();) {
    if (!RangesOverlap(ptr, size, (*it)->ptr(), (*it)->size())) {
      ++it;
      continue;
    }
    if (!predicate(*it)) {
      ok = false;
      ++it;
      continue;
    }
    allocations_by_ptr_.erase(Begin(*it));
    it = allocations_.erase(it);
  }
  return ok;
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

bool VMMAutoGrowthBestFitBlockAllocationV2::SetVMMRemapEvent(
    gpuStream_t stream, std::shared_ptr<CUDAEventGuard> event) {
  if (owner_ == nullptr) {
    return false;
  }
  remap_stream_ = stream;
  remap_event_ = std::move(event);
  return true;
}

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
  // the driver-level allocator throws EnforceNotMet. Convert it to BadAlloc
  // so outer allocator layers can run their generic OOM recovery path.
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

  return new VMMAutoGrowthBestFitBlockAllocationV2(  // NOLINT
      it,
      place_,
      this);
}

size_t VMMAutoGrowthBestFitAllocatorV2::CompactImpl(const Place& place,
                                                    size_t requested_size) {
  const uint64_t compact_seq =
      g_vmm_v2_compact_seq.fetch_add(1, std::memory_order_relaxed) + 1;
  const auto compact_start = Clock::now();
  // Defensive place validation: the call chain
  // (RetryAllocator -> StreamSafe -> MultiPool -> SinglePool) guarantees
  // place consistency.  Log a warning on mismatch but do not throw,
  // since CompactImpl is called inside a try-catch that would silently
  // swallow the exception and skip compaction.
  if (UNLIKELY(place != Place(place_))) {
    LOG(WARNING) << "CompactImpl place mismatch: got " << place.DebugString()
                 << " but allocator serves " << Place(place_).DebugString();
  }
  const auto lock_start = Clock::now();
  std::lock_guard<SpinLock> guard(spinlock_);
  const uint64_t lock_wait_us = ElapsedMicros(lock_start, Clock::now());

  const auto block_scan_start = Clock::now();
  size_t total_free = 0;
  size_t max_free = 0;
  size_t tail_free = 0;
  size_t mapped_free_blocks = 0;
  size_t mapped_free_bytes = 0;
  size_t largest_mapped_free = 0;
  size_t indexable_mapped_free_blocks = 0;
  size_t indexable_mapped_free_bytes = 0;
  size_t unmapped_free_blocks_count = 0;
  size_t unmapped_free_bytes = 0;
  size_t largest_unmapped_free = 0;
  std::vector<std::pair<VMMDevicePtr, size_t>> compact_source_ranges;
  for (const auto& blk : all_blocks_) {
    if (blk.IsMappedFree()) {
      ++mapped_free_blocks;
      mapped_free_bytes += blk.size_;
      largest_mapped_free = std::max(largest_mapped_free, blk.size_);
      total_free += blk.size_;
      compact_source_ranges.emplace_back(blk.va_range());
    }
    if (CanIndexFreeBlock(blk)) {
      ++indexable_mapped_free_blocks;
      indexable_mapped_free_bytes += blk.size_;
      max_free = std::max(max_free, blk.size_);
    }
    if (blk.IsUnmappedFree()) {
      ++unmapped_free_blocks_count;
      unmapped_free_bytes += blk.size_;
      largest_unmapped_free = std::max(largest_unmapped_free, blk.size_);
    }
  }
  const bool has_tail_block = !all_blocks_.empty();
  const bool tail_is_indexable =
      has_tail_block && CanIndexFreeBlock(all_blocks_.back());
  const bool tail_is_unmapped =
      has_tail_block && all_blocks_.back().IsUnmappedFree();
  if (tail_is_indexable) {
    tail_free = all_blocks_.back().size_;
  }
  const uint64_t block_scan_us = ElapsedMicros(block_scan_start, Clock::now());
  uint64_t source_precheck_us = 0;
  uint64_t driver_topup_us = 0;

  auto log_compact_precheck_summary =
      [&](const char* reason,
          size_t current_compact_target,
          size_t current_required_releasable_bytes,
          size_t current_releasable_target_bytes,
          size_t current_releasable_handles,
          size_t current_releasable_bytes,
          size_t bounded_source_page_count,
          const DriverMemoryStats* driver_memory_stats = nullptr) {
        const size_t missing_releasable_bytes =
            current_required_releasable_bytes > current_releasable_bytes
                ? current_required_releasable_bytes - current_releasable_bytes
                : 0;
        const size_t driver_topup_gap_bytes =
            driver_memory_stats != nullptr &&
                    missing_releasable_bytes >
                        driver_memory_stats->driver_actual_avail
                ? missing_releasable_bytes -
                      driver_memory_stats->driver_actual_avail
                : 0;

        VLOG(3)
            << "VMM V2 compact skip: seq=" << compact_seq
            << " pool=" << static_cast<int>(pool_type_) << " action=skip"
            << " reason=" << reason << " requested=" << requested_size
            << " compact_target=" << current_compact_target
            << " required_releasable_bytes="
            << current_required_releasable_bytes
            << " releasable_target_bytes=" << current_releasable_target_bytes
            << " releasable_handles=" << current_releasable_handles
            << " releasable_bytes=" << current_releasable_bytes
            << " missing_releasable_bytes=" << missing_releasable_bytes
            << " bounded_source_pages=" << bounded_source_page_count
            << " source_ranges=" << compact_source_ranges.size()
            << " mapped_free_blocks=" << mapped_free_blocks
            << " mapped_free_bytes=" << mapped_free_bytes
            << " largest_mapped_free=" << largest_mapped_free
            << " indexable_mapped_free_blocks=" << indexable_mapped_free_blocks
            << " indexable_mapped_free_bytes=" << indexable_mapped_free_bytes
            << " largest_indexable_mapped_free=" << max_free
            << " unmapped_free_blocks=" << unmapped_free_blocks_count
            << " unmapped_free_bytes=" << unmapped_free_bytes
            << " largest_unmapped_free=" << largest_unmapped_free
            << " tail_free=" << tail_free
            << " tail_is_indexable=" << tail_is_indexable
            << " tail_is_unmapped=" << tail_is_unmapped
            << " all_blocks=" << all_blocks_.size()
            << " free_index_size=" << free_blocks_.size()
            << " unmapped_free_index_size=" << unmapped_free_blocks_.size()
            << " driver_actual_avail="
            << (driver_memory_stats == nullptr
                    ? 0UL
                    : driver_memory_stats->driver_actual_avail)
            << " driver_actual_total="
            << (driver_memory_stats == nullptr
                    ? 0UL
                    : driver_memory_stats->driver_actual_total)
            << " driver_topup_gap_bytes=" << driver_topup_gap_bytes
            << " total_us=" << ElapsedMicros(compact_start, Clock::now())
            << " lock_wait_us=" << lock_wait_us
            << " block_scan_us=" << block_scan_us
            << " source_precheck_us=" << source_precheck_us
            << " driver_topup_us=" << driver_topup_us
            << " driver_mem_info_collected=" << (driver_memory_stats != nullptr)
            << " mem_info_status="
            << static_cast<int>(driver_memory_stats == nullptr
                                    ? phi::gpuSuccess
                                    : driver_memory_stats->mem_info_status);
      };

  size_t compact_target = requested_size;
  if (requested_size > 0) {
    if (max_free >= requested_size) {
      log_compact_precheck_summary(
          "large_free_block_available", compact_target, 0, 0, 0, 0, 0);
      VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
              << " compact skip: max_free=" << max_free
              << " >= requested=" << requested_size;
      return 0;
    }

    if (total_free < requested_size) {
      if (total_free <= tail_free) {
        log_compact_precheck_summary(
            "insufficient_non_tail_mapped_free", compact_target, 0, 0, 0, 0, 0);
        VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
                << " compact skip: total_free=" << total_free
                << " < requested=" << requested_size
                << " and no non-tail free bytes are available"
                << " (tail_free=" << tail_free << ")";
        return 0;
      }
      // Partial compact: under tight training pressure, mapped-free bytes may
      // be insufficient to cover the whole request but still reduce the next
      // grow attempt. Move the non-tail free backing to tail/gaps and let the
      // following allocation retry grow only the remaining deficit.
      compact_target = total_free;
      VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
              << " compact partial: total_free=" << total_free
              << " < requested=" << requested_size << " tail_free=" << tail_free
              << " compact_target=" << compact_target;
    }
  }

  // Count potentially movable source pages through the BackingMap mirror.
  // Runtime event readiness remains in RemapTransaction; this precheck only
  // verifies that free VA ranges fully cover mapped, reusable backing pages.
  // The source ranges include all mapped-free blocks. BackingMap page state
  // decides which individual handles are movable.
  const size_t required_releasable_bytes =
      compact_target > tail_free ? compact_target - tail_free : 0;
  const size_t releasable_target_bytes =
      requested_size > 0 ? required_releasable_bytes : compact_target;

  const auto source_precheck_start = Clock::now();
  auto source_pages = underlying_allocator_->CollectRemapSourcePages(
      compact_source_ranges, releasable_target_bytes);
  size_t releasable_handles = 0;
  for (const auto& page : source_pages) {
    if (page.remap_source_state == VMMBackingMap::RemapSourceState::kReady) {
      ++releasable_handles;
    }
  }
  source_precheck_us = ElapsedMicros(source_precheck_start, Clock::now());
  const size_t releasable_bytes =
      releasable_handles * underlying_allocator_->handle_size();
  DriverMemoryStats topup_memory_stats;
  const DriverMemoryStats* topup_memory_stats_ptr = nullptr;

  if (requested_size > 0 && releasable_bytes < required_releasable_bytes) {
    const auto driver_topup_start = Clock::now();
    topup_memory_stats = CollectDriverMemoryStats(place_.device);
    topup_memory_stats_ptr = &topup_memory_stats;
    driver_topup_us = ElapsedMicros(driver_topup_start, Clock::now());
    const size_t driver_topup_bytes =
        required_releasable_bytes - releasable_bytes;
    const bool driver_can_top_up =
        topup_memory_stats.mem_info_status == phi::gpuSuccess &&
        topup_memory_stats.driver_actual_avail >= driver_topup_bytes;
    if (!driver_can_top_up) {
      log_compact_precheck_summary("insufficient_releasable_mapped_free",
                                   compact_target,
                                   required_releasable_bytes,
                                   releasable_target_bytes,
                                   releasable_handles,
                                   releasable_bytes,
                                   source_pages.size(),
                                   &topup_memory_stats);
      VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
              << " compact skip: releasable_bytes=" << releasable_bytes
              << " < required=" << required_releasable_bytes
              << " and driver_actual_avail="
              << topup_memory_stats.driver_actual_avail
              << " < driver_topup_bytes=" << driver_topup_bytes
              << " requested=" << requested_size << " total_free=" << total_free
              << " max_free=" << max_free << " tail_free=" << tail_free
              << " compact_target=" << compact_target
              << " source_ranges=" << compact_source_ranges.size();
      return 0;
    }
    compact_target = releasable_bytes;
    VLOG(3) << "VMM V2 pool " << static_cast<int>(pool_type_)
            << " compact partial with driver top-up: releasable_bytes="
            << releasable_bytes
            << " required_releasable_bytes=" << required_releasable_bytes
            << " driver_actual_avail=" << topup_memory_stats.driver_actual_avail
            << " driver_topup_bytes=" << driver_topup_bytes
            << " requested=" << requested_size
            << " compact_target=" << compact_target;
  }

  if (releasable_handles == 0) {
    log_compact_precheck_summary("no_releasable_handles",
                                 compact_target,
                                 required_releasable_bytes,
                                 releasable_target_bytes,
                                 releasable_handles,
                                 releasable_bytes,
                                 source_pages.size(),
                                 topup_memory_stats_ptr);
    VLOG(4) << "VMM V2 pool " << static_cast<int>(pool_type_)
            << " compact skip: no releasable handles"
            << " (total_free=" << total_free << " max_free=" << max_free
            << " tail_free=" << tail_free << " requested=" << requested_size
            << " compact_target=" << compact_target
            << " releasable_handles=" << releasable_handles
            << " releasable_bytes=" << releasable_bytes
            << " source_ranges=" << compact_source_ranges.size() << ")";
    return 0;
  }

  const size_t driver_topup_bytes =
      required_releasable_bytes > releasable_bytes
          ? required_releasable_bytes - releasable_bytes
          : 0;
  VLOG(3) << "VMM V2 compact attempt: seq=" << compact_seq
          << " pool=" << static_cast<int>(pool_type_)
          << " requested=" << requested_size
          << " compact_target=" << compact_target
          << " partial=" << (compact_target < requested_size)
          << " total_free=" << total_free << " max_free=" << max_free
          << " tail_free=" << tail_free
          << " required_releasable_bytes=" << required_releasable_bytes
          << " releasable_handles=" << releasable_handles
          << " releasable_bytes=" << releasable_bytes
          << " driver_topup_bytes=" << driver_topup_bytes
          << " source_pages=" << source_pages.size();

  auto commit_synthetic_allocation = [this](DecoratedAllocationPtr allocation) {
    TrackUnderlyingAllocation(std::move(allocation));
  };
  auto can_use_destination_range = [this](void* ptr, size_t size) {
    return CanPrepareRemapDestinationRange(ptr, size);
  };
  auto release_stale_destination_allocations = [this](void* ptr, size_t size) {
    return PrepareRemapDestinationRange(ptr, size);
  };
  FreeBlockRemapCompactor compactor(underlying_allocator_,
                                    pool_type_,
                                    commit_synthetic_allocation,
                                    can_use_destination_range,
                                    release_stale_destination_allocations);
  const size_t remap_target = requested_size > 0 ? compact_target : 0;
  const auto compactor_start = Clock::now();
  const size_t remapped =
      compactor.Compact(&all_blocks_, remap_target, compact_seq);
  const uint64_t compactor_us = ElapsedMicros(compactor_start, Clock::now());
  // Always rebuild: Phase 1 may have replaced FREE blocks with
  // UNMAPPED-FREE/FREE
  // segments before Phase 2 fails.  Without rebuild, free_blocks_ holds
  // stale iterators to erased list nodes, causing use-after-free on next alloc.
  const auto rebuild_index_start = Clock::now();
  RebuildFreeBlockIndex();
  const uint64_t rebuild_index_us =
      ElapsedMicros(rebuild_index_start, Clock::now());
  VLOG(3) << "VMM V2 compact finish: seq=" << compact_seq
          << " pool=" << static_cast<int>(pool_type_)
          << " requested=" << requested_size << " remapped_bytes=" << remapped
          << " total_us=" << ElapsedMicros(compact_start, Clock::now())
          << " compactor_us=" << compactor_us
          << " rebuild_index_us=" << rebuild_index_us;
  return remapped;
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
  auto remap_event = wrapped_allocation->TakeRemapEvent();
  if (remap_event != nullptr) {
    PADDLE_ENFORCE_EQ(
        underlying_allocator_->SetBlockRemapEvent(
            *it, wrapped_allocation->remap_stream(), remap_event),
        true,
        common::errors::InvalidArgument(
            "Failed to attach explicit VMM V2 remap event for block %p.",
            it->ptr_));
  } else {
    it->SetRemapSafety(wrapped_allocation->remap_stream(), nullptr);
  }
  it->MarkFree();
  TryMerge(it);
  delete allocation;
}

void VMMAutoGrowthBestFitAllocatorV2::GetFreeBlockStats(size_t* total_free,
                                                        size_t* max_free) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t total = 0;
  for (const auto& entry : free_blocks_) {
    total += entry.first.first;
  }
  size_t max_sz = 0;
  if (!free_blocks_.empty()) {
    max_sz = free_blocks_.rbegin()->first.first;
  }
  *total_free = total;
  *max_free = max_sz;
}

bool VMMAutoGrowthBestFitAllocatorV2::CollectTensorParts(
    void* ptr,
    size_t size,
    std::vector<BlockPart>* parts,
    bool mark_ipc_exported) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto target_va = reinterpret_cast<VMMDevicePtr>(ptr);
  PADDLE_ENFORCE_LE(
      size,
      std::numeric_limits<VMMDevicePtr>::max() - target_va,
      common::errors::InvalidArgument(
          "Invalid VMM V2 tensor range: ptr %p plus size %zu overflows.",
          ptr,
          size));
  BlockListIt block_it = all_blocks_.end();
  for (auto it = all_blocks_.begin(); it != all_blocks_.end(); ++it) {
    if (!it->IsActive()) {
      continue;
    }
    if (it->ContainsVARange(target_va, size)) {
      block_it = it;
      break;
    }
  }
  if (block_it == all_blocks_.end()) {
    VLOG(8) << "[VMM-IPC/export] VMM v2 best-fit no active block for "
            << "target_ptr=" << ptr << " target_size=" << size
            << " pool=" << static_cast<int>(pool_type_)
            << " block_count=" << all_blocks_.size();
    return false;
  }

  std::vector<BlockPart> collected;
  auto collect_ipc_parts = [&] {
    return underlying_allocator_->CollectIPCParts(
        target_va, size, parts != nullptr ? &collected : nullptr);
  };
  if (!collect_ipc_parts()) {
    const size_t cleared =
        underlying_allocator_->ClearRemapDestinationOwnershipInRange(target_va,
                                                                     size);
    if (cleared > 0) {
      collected.clear();
    }
    if (cleared == 0 || !collect_ipc_parts()) {
      VLOG(8) << "[VMM-IPC/export] VMM v2 best-fit failed to collect backing "
              << "parts for active block ptr=" << block_it->ptr_
              << " block_size=" << block_it->size_ << " target_ptr=" << ptr
              << " target_size=" << size
              << " pool=" << static_cast<int>(pool_type_);
      return false;
    }
  }
  if (mark_ipc_exported) {
    if (!underlying_allocator_->MarkIPCExported(target_va, size)) {
      VLOG(8) << "[VMM-IPC/export] VMM v2 best-fit failed to mark IPC exported "
              << "for active block ptr=" << block_it->ptr_
              << " block_size=" << block_it->size_ << " target_ptr=" << ptr
              << " target_size=" << size
              << " pool=" << static_cast<int>(pool_type_);
      return false;
    }
    block_it->ipc_exported_ = true;
  }
  if (parts != nullptr) {
    *parts = std::move(collected);
  }
  return true;
}

bool VMMAutoGrowthBestFitAllocatorV2::SetBlockRemapEvent(
    void* ptr, gpuStream_t stream, std::shared_ptr<CUDAEventGuard> event) {
  std::lock_guard<SpinLock> guard(spinlock_);
  for (auto it = all_blocks_.begin(); it != all_blocks_.end(); ++it) {
    if (!it->IsActive() || it->ptr_ != ptr) {
      continue;
    }
    return underlying_allocator_->SetBlockRemapEvent(
        *it, stream, std::move(event));
  }
  return false;
}

bool VMMAutoGrowthBestFitAllocatorV2::SetBlockRemapEvent(
    BlockListIt block_it,
    gpuStream_t stream,
    std::shared_ptr<CUDAEventGuard> event) {
  std::lock_guard<SpinLock> guard(spinlock_);
  if (block_it == all_blocks_.end() || !block_it->IsActive()) {
    return false;
  }
  return underlying_allocator_->SetBlockRemapEvent(
      *block_it, stream, std::move(event));
}

BlockList VMMAutoGrowthBestFitAllocatorV2::SnapshotAllBlocks() const {
  std::lock_guard<SpinLock> guard(spinlock_);
  return all_blocks_;
}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocFromFreeBlocks(
    size_t size) {
  auto it = free_blocks_.lower_bound({size, nullptr});
  while (it != free_blocks_.end() && !CanIndexFreeBlock(*it->second)) {
    it = free_blocks_.erase(it);
  }
  if (it == free_blocks_.end()) {
    return nullptr;
  }

  auto block_it = it->second;
  EraseFreeBlock(block_it);

  if (block_it->size_ > size) {
    const size_t remaining_size = block_it->size_ - size;
    BlockV2 remaining_block =
        block_it->MakeMappedFreeSubBlock(size, remaining_size);
    // The free remainder keeps the source block's remap-safety stream. The
    // reused prefix is cleared by MarkActive().

    block_it->TrimToPrefix(size);
    auto remain_it =
        all_blocks_.insert(std::next(block_it), std::move(remaining_block));
    InsertFreeBlock(remain_it);
  }

  block_it->MarkActive();
  return new VMMAutoGrowthBestFitBlockAllocationV2(  // NOLINT
      block_it,
      place_,
      this);  // NOLINT
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
    if (underlying_allocations_.Overlaps(it->ptr_, backing_size)) {
      VLOG(6) << "VMM V2 AllocFromUnmappedFreeBlocks skip "
                 "underlying-overlapped unmapped-free ptr="
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
    mapped_remain.owning_stream_ = nullptr;
    mapped_remain.remap_safe_event_.reset();
    mapped_remain.remap_pending_states_.clear();
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

  return new VMMAutoGrowthBestFitBlockAllocationV2(  // NOLINT
      best,
      place_,
      this);  // NOLINT
}

void VMMAutoGrowthBestFitAllocatorV2::TrackUnderlyingAllocation(
    DecoratedAllocationPtr allocation) {
  underlying_allocations_.Add(std::move(allocation));
}

bool VMMAutoGrowthBestFitAllocatorV2::AllocationOwnedByRemapDestination(
    const DecoratedAllocationPtr& allocation,
    void* target_ptr,
    size_t target_size) const {
  if (!underlying_allocator_->IsAllocationOwnedByRemapDestination(
          allocation->ptr())) {
    VLOG(6) << "VMM V2 synthetic allocation preparation: target range "
            << target_ptr << " size=" << target_size
            << " overlaps non-remap-destination underlying allocation "
            << allocation->ptr() << " size=" << allocation->size();
    return false;
  }
  return true;
}

bool VMMAutoGrowthBestFitAllocatorV2::CanPrepareRemapDestinationRange(
    void* ptr, size_t size) const {
  if (underlying_allocations_.AllOverlapsSatisfy(
          ptr,
          size,
          [this, ptr, size](const DecoratedAllocationPtr& allocation) {
            return AllocationOwnedByRemapDestination(allocation, ptr, size);
          })) {
    return true;
  }
  return underlying_allocator_->IsRangeUnmapped(
      reinterpret_cast<VMMDevicePtr>(ptr), size);
}

bool VMMAutoGrowthBestFitAllocatorV2::PrepareRemapDestinationRange(
    void* ptr, size_t size) {
  const bool released_owned_overlaps = underlying_allocations_.EraseOverlapsIf(
      ptr, size, [this, ptr, size](const DecoratedAllocationPtr& allocation) {
        if (!AllocationOwnedByRemapDestination(allocation, ptr, size)) {
          return false;
        }
        VLOG(3) << "VMM V2 synthetic allocation preparation: releasing stale "
                   "remap-destination allocation "
                << allocation->ptr() << " size=" << allocation->size();
        return true;
      });
  if (released_owned_overlaps) {
    return true;
  }
  return underlying_allocator_->IsRangeUnmapped(
      reinterpret_cast<VMMDevicePtr>(ptr), size);
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

bool VMMAutoGrowthBestFitAllocatorV2::CanReleaseIdleUnderlying(
    uint8_t* base, size_t size) const {
  if (!IsRangeEntirelyFree(base, size)) {
    return false;
  }
  return underlying_allocator_->IsRangeReleasable(
      reinterpret_cast<VMMDevicePtr>(base), size);
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

bool VMMAutoGrowthBestFitAllocatorV2::TryReleaseIdleUnderlying(
    UnderlyingAllocationRegistry::iterator* alloc_it, uint64_t* released) {
  auto* allocation = (**alloc_it).get();
  auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
  const size_t alloc_size = allocation->size();
  if (!CanReleaseIdleUnderlying(base, alloc_size)) {
    return false;
  }

  if (underlying_allocator_->IsAllocationOwnedByRemapDestination(base)) {
    const size_t cleared =
        underlying_allocator_->ClearRemapDestinationOwnershipInRange(
            reinterpret_cast<VMMDevicePtr>(base), alloc_size);
    VLOG(5) << "VMM V2 pool " << static_cast<int>(pool_type_)
            << " cleared remap-destination ownership for " << cleared
            << " bytes before releasing idle chunk " << base
            << " size=" << alloc_size;
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

void VMMAutoGrowthBestFitAllocatorV2::RebuildFreeBlockIndex() {
  free_blocks_.clear();
  unmapped_free_blocks_.clear();
  for (auto it = all_blocks_.begin(); it != all_blocks_.end(); ++it) {
    if (CanIndexFreeBlock(*it)) {
      InsertFreeBlock(it);
    }
    if (it->IsUnmappedFree()) {
      InsertUnmappedFreeBlock(it);
    }
  }
}

void VMMAutoGrowthBestFitAllocatorV2::TryMerge(BlockListIt it) {
  // Only adjacent FREE blocks are merged here. ACTIVE blocks are never touched,
  // and unmapped-free blocks remain as explicit holes for later remap/reuse.
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

  if (CanIndexFreeBlock(*it)) {
    InsertFreeBlock(it);
  }
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
  // Returns true when the range contains only FREE/unmapped-free blocks or
  // when
  // blocks have already been removed by a prior FreeIdleChunks pass
  // (unmapped-free scatter / single-unmapped-free path case: the original
  // allocation's cleanup removes
  // blocks in the overlapping VA range before the synthetic allocation
  // is processed).  FreeImpl handles this safely: original allocation
  // skips remapped handles; synthetic allocation unmaps+releases its own.
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
      const bool unmapped_free = it->IsUnmappedFree();
      BlockV2 right =
          unmapped_free ? it->MakeUnmappedFreeSubBlock(right_offset, right_size)
                        : it->MakeMappedFreeSubBlock(right_offset, right_size);
      if (!unmapped_free) {
        right.CopyRemapSafetyFrom(*it);
      }

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
