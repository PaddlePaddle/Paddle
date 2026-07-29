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
#include "paddle/phi/core/scope_guard.h"

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

size_t FullyCoveredBackingBytes(const std::pair<VMMDevicePtr, size_t>& range,
                                VMMDevicePtr virtual_mem_base,
                                size_t handle_size) {
  if (handle_size == 0) {
    return 0;
  }
  const size_t remainder = (range.first - virtual_mem_base) % handle_size;
  const size_t prefix = remainder == 0 ? 0 : handle_size - remainder;
  if (range.second <= prefix) {
    return 0;
  }
  return ((range.second - prefix) / handle_size) * handle_size;
}

void PrioritizeContiguousSourceRanges(
    std::vector<std::pair<VMMDevicePtr, size_t>>* ranges,
    VMMDevicePtr virtual_mem_base,
    size_t handle_size) {
  std::stable_sort(
      ranges->begin(),
      ranges->end(),
      [virtual_mem_base, handle_size](const auto& lhs, const auto& rhs) {
        return FullyCoveredBackingBytes(lhs, virtual_mem_base, handle_size) >
               FullyCoveredBackingBytes(rhs, virtual_mem_base, handle_size);
      });
}

}  // namespace

struct VMMAutoGrowthBestFitAllocatorV2::CompactState {
  size_t total_free{0};
  size_t max_free{0};
  size_t tail_free{0};
  std::vector<std::pair<VMMDevicePtr, size_t>> source_ranges;
};

struct VMMAutoGrowthBestFitAllocatorV2::CompactContext {
  uint64_t seq{0};
  Clock::time_point start;
  size_t requested_size{0};
  size_t compact_target{0};
  size_t ready_bytes{0};
  uint64_t lock_wait_us{0};
  uint64_t block_scan_us{0};
  uint64_t source_precheck_us{0};
  uint64_t driver_topup_us{0};
  bool driver_memory_collected{false};
  DriverMemoryStats driver_memory;
};

void VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::Add(
    DecoratedAllocationPtr* allocation) {
  PADDLE_ENFORCE_NOT_NULL(
      allocation,
      common::errors::InvalidArgument(
          "VMM V2 registry requires a non-null ownership slot."));
  PADDLE_ENFORCE_NOT_NULL(
      allocation->get(),
      common::errors::InvalidArgument(
          "VMM V2 registry cannot add an empty allocation."));
  allocations_.emplace_back(std::move(*allocation));
  auto it = std::prev(allocations_.end());
  auto* begin = Begin(*it);
  try {
    PADDLE_ENFORCE_EQ(
        allocations_by_ptr_.emplace(begin, it).second,
        true,
        common::errors::AlreadyExists(
            "Duplicate underlying allocation base %p in VMM V2 registry.",
            begin));
  } catch (...) {
    *allocation = std::move(*it);
    allocations_.erase(it);
    throw;
  }
}

void VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::
    AddAllOrRestore(std::vector<DecoratedAllocationPtr>* allocations) {
  PADDLE_ENFORCE_NOT_NULL(
      allocations,
      common::errors::InvalidArgument(
          "VMM V2 registry requires a non-null allocation batch."));
  std::vector<uint8_t*> added_bases;
  added_bases.reserve(allocations->size());
  try {
    for (auto& allocation : *allocations) {
      auto* base = Begin(allocation);
      Add(&allocation);
      added_bases.push_back(base);
    }
  } catch (...) {
    for (size_t i = 0; i < added_bases.size(); ++i) {
      auto index_it = allocations_by_ptr_.find(added_bases[i]);
      PADDLE_ENFORCE_NE(
          index_it,
          allocations_by_ptr_.end(),
          common::errors::PreconditionNotMet(
              "VMM V2 registry lost allocation %p during batch rollback.",
              added_bases[i]));
      auto list_it = index_it->second;
      (*allocations)[i] = std::move(*list_it);
      allocations_by_ptr_.erase(index_it);
      allocations_.erase(list_it);
    }
    throw;
  }
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

size_t RangeOverlapBytes(void* lhs_ptr,
                         size_t lhs_size,
                         void* rhs_ptr,
                         size_t rhs_size) {
  const auto lhs_begin = reinterpret_cast<uintptr_t>(lhs_ptr);
  const auto lhs_end = lhs_begin + lhs_size;
  const auto rhs_begin = reinterpret_cast<uintptr_t>(rhs_ptr);
  const auto rhs_end = rhs_begin + rhs_size;
  if (lhs_end <= rhs_begin || rhs_end <= lhs_begin) {
    return 0;
  }
  return std::min(lhs_end, rhs_end) - std::max(lhs_begin, rhs_begin);
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
VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::FindByAddress(
    VMMDevicePtr ptr) {
  auto index_it = allocations_by_ptr_.find(reinterpret_cast<uint8_t*>(ptr));
  return index_it == allocations_by_ptr_.end() ? allocations_.end()
                                               : index_it->second;
}

DecoratedAllocationPtr
VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::Take(
    iterator it) {
  allocations_by_ptr_.erase(Begin(*it));
  DecoratedAllocationPtr allocation = std::move(*it);
  allocations_.erase(it);
  return allocation;
}

VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::iterator
VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::Erase(
    iterator it) {
  allocations_by_ptr_.erase(Begin(*it));
  return allocations_.erase(it);
}

VMMAutoGrowthBestFitAllocatorV2::UnderlyingRanges
VMMAutoGrowthBestFitAllocatorV2::UnderlyingAllocationRegistry::
    CollectRangesByAddress() const {
  UnderlyingRanges ranges;
  ranges.reserve(allocations_by_ptr_.size());
  for (const auto& [base, allocation_it] : allocations_by_ptr_) {
    ranges.emplace_back(reinterpret_cast<VMMDevicePtr>(base),
                        (*allocation_it)->size());
  }
  return ranges;
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
  CUDAVirtualMemAllocatorV2::AllocationWithBlock grow_alloc;
  if (grow_size > 0) {
    bool grow_completed = false;
    DEFINE_PADDLE_SCOPE_GUARD([&] {
      if (!grow_completed) {
        restore_tail_free_block();
      }
    });
    grow_alloc = underlying_allocator_->AppendWithBlock(grow_size);
    grow_completed = true;
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

VMMAutoGrowthBestFitAllocatorV2::CompactState
VMMAutoGrowthBestFitAllocatorV2::CollectCompactState() {
  CompactState state;
  const BlockV2* tail_free_block =
      !all_blocks_.empty() && CanIndexFreeBlock(all_blocks_.back())
          ? &all_blocks_.back()
          : nullptr;
  for (auto& block : all_blocks_) {
    if (block.IsMappedFree()) {
      state.total_free += block.size_;
      // Tail free already contributes to the contiguous allocation target.
      // Moving it only shifts the same free space to a higher VA.
      if (&block != tail_free_block &&
          RemapTransaction::CheckBlockRemapSafety(&block)) {
        state.source_ranges.emplace_back(block.va_range());
      }
    }
    if (CanIndexFreeBlock(block)) {
      state.max_free = std::max(state.max_free, block.size_);
    }
  }

  if (tail_free_block != nullptr) {
    state.tail_free = all_blocks_.back().size_;
  }
  return state;
}

void VMMAutoGrowthBestFitAllocatorV2::LogCompactSkip(
    const CompactState& state,
    const CompactContext& context,
    const char* reason) const {
  VLOG(3) << "VMM V2 compact skip: seq=" << context.seq
          << " pool=" << static_cast<int>(pool_type_) << " action=skip"
          << " reason=" << reason << " requested=" << context.requested_size
          << " compact_target=" << context.compact_target
          << " total_free=" << state.total_free
          << " max_free=" << state.max_free << " tail_free=" << state.tail_free
          << " ready_bytes=" << context.ready_bytes
          << " source_ranges=" << state.source_ranges.size()
          << " driver_actual_avail="
          << (context.driver_memory_collected
                  ? context.driver_memory.driver_actual_avail
                  : 0UL)
          << " driver_actual_total="
          << (context.driver_memory_collected
                  ? context.driver_memory.driver_actual_total
                  : 0UL)
          << " total_us=" << ElapsedMicros(context.start, Clock::now())
          << " lock_wait_us=" << context.lock_wait_us
          << " block_scan_us=" << context.block_scan_us
          << " source_precheck_us=" << context.source_precheck_us
          << " driver_topup_us=" << context.driver_topup_us
          << " driver_mem_info_collected=" << context.driver_memory_collected
          << " mem_info_status="
          << static_cast<int>(context.driver_memory_collected
                                  ? context.driver_memory.mem_info_status
                                  : phi::gpuSuccess);
}

size_t VMMAutoGrowthBestFitAllocatorV2::CompactImpl(const Place& place) {
  return RemapForAllocation(place, 0);
}

size_t VMMAutoGrowthBestFitAllocatorV2::RemapForAllocation(
    const Place& place,
    size_t requested_size,
    const VMMGrowOOMInfo* grow_oom,
    VMMRemapAttemptResult* attempt_result) {
  if (attempt_result != nullptr) {
    *attempt_result = VMMRemapAttemptResult{};
  }
  const uint64_t compact_seq =
      g_vmm_v2_compact_seq.fetch_add(1, std::memory_order_relaxed) + 1;
  const auto compact_start = Clock::now();
  PADDLE_ENFORCE_EQ(
      place,
      Place(place_),
      common::errors::InvalidArgument(
          "VMM V2 compact place %s does not match allocator place %s.",
          place.DebugString(),
          Place(place_).DebugString()));
  platform::CUDADeviceGuard device_guard(place_.device);
  const auto lock_start = Clock::now();
  std::lock_guard<SpinLock> guard(spinlock_);
  const uint64_t lock_wait_us = ElapsedMicros(lock_start, Clock::now());

  const auto block_scan_start = Clock::now();
  CompactState compact_state = CollectCompactState();
  const uint64_t block_scan_us = ElapsedMicros(block_scan_start, Clock::now());
  CompactContext compact_context;
  compact_context.seq = compact_seq;
  compact_context.start = compact_start;
  compact_context.requested_size = requested_size;
  compact_context.lock_wait_us = lock_wait_us;
  compact_context.block_scan_us = block_scan_us;

  size_t compact_target = requested_size;
  compact_context.compact_target = compact_target;
  const size_t aligned_request =
      requested_size > 0 ? AlignedSize(requested_size, alignment_) : 0;
  bool use_grow_oom = false;
  if (requested_size > 0) {
    if (compact_state.max_free >= aligned_request) {
      LogCompactSkip(
          compact_state, compact_context, "large_free_block_available");
      if (attempt_result != nullptr) {
        attempt_result->status = VMMRemapAttemptStatus::kRetryWithoutRemap;
      }
      return 0;
    }

    if (compact_state.total_free <= compact_state.tail_free) {
      LogCompactSkip(
          compact_state, compact_context, "insufficient_non_tail_mapped_free");
      if (attempt_result != nullptr) {
        attempt_result->status = VMMRemapAttemptStatus::kNoMovableMemory;
      }
      return 0;
    }

    if (grow_oom != nullptr) {
      const size_t handle_size = underlying_allocator_->handle_size();
      const size_t grow_bytes = aligned_request > compact_state.tail_free
                                    ? aligned_request - compact_state.tail_free
                                    : 0;
      const size_t expected_grow_handles =
          grow_bytes / handle_size + (grow_bytes % handle_size != 0);
      use_grow_oom = grow_oom->device == place_.device &&
                     grow_oom->pool_type == pool_type_ &&
                     grow_oom->handle_size == handle_size &&
                     grow_oom->requested_handles == expected_grow_handles &&
                     grow_oom->created_handles < grow_oom->requested_handles;
      if (use_grow_oom) {
        const size_t required_handles =
            grow_oom->requested_handles - grow_oom->created_handles;
        compact_target =
            compact_state.tail_free + required_handles * handle_size;
        compact_context.compact_target = compact_target;
      }
    }
  }

  if (requested_size > 0) {
    // Prefer obtaining the bounded source budget from as few contiguous free
    // ranges as possible. This reduces the number of source VA gaps without
    // adding work to successful allocation/free paths.
    PrioritizeContiguousSourceRanges(&compact_state.source_ranges,
                                     underlying_allocator_->virtual_mem_base(),
                                     underlying_allocator_->handle_size());
  }

  // Count potentially movable source pages through the BackingMap mirror.
  // Runtime event readiness remains in RemapTransaction; this precheck only
  // verifies that free VA ranges fully cover mapped, reusable backing pages.
  // Source ranges exclude reusable tail free because it already contributes
  // to the contiguous destination. BackingMap page state decides which
  // remaining handles are movable.
  const size_t required_releasable_bytes =
      compact_target > compact_state.tail_free
          ? compact_target - compact_state.tail_free
          : 0;
  const size_t releasable_target_bytes =
      requested_size > 0 ? required_releasable_bytes : compact_target;
  size_t remap_target = releasable_target_bytes;
  const auto source_precheck_start = Clock::now();
  auto source_pages = underlying_allocator_->CollectRemapSourcePages(
      compact_state.source_ranges, releasable_target_bytes);
  size_t releasable_handles = 0;
  for (const auto& page : source_pages) {
    if (page.remap_source_state == VMMBackingMap::RemapSourceState::kReady) {
      ++releasable_handles;
    }
  }
  compact_context.source_precheck_us =
      ElapsedMicros(source_precheck_start, Clock::now());
  size_t releasable_bytes =
      releasable_handles * underlying_allocator_->handle_size();
  compact_context.ready_bytes = releasable_bytes;
  if (attempt_result != nullptr) {
    attempt_result->movable_bytes = releasable_bytes;
    attempt_result->required_bytes = required_releasable_bytes;
  }

  if (use_grow_oom && releasable_bytes < required_releasable_bytes) {
    // A training allocation must either recover immediately or leave the
    // allocator unchanged. The failed grow already measured how many handles
    // the driver could create; moving fewer than the remaining deficit cannot
    // make the retry succeed. RetryAllocator will reevaluate after a real free
    // notification instead of paying for speculative remap work here.
    LogCompactSkip(compact_state,
                   compact_context,
                   "insufficient_releasable_for_failed_grow");
    if (attempt_result != nullptr) {
      attempt_result->status =
          VMMRemapAttemptStatus::kInsufficientMovableMemory;
    }
    return 0;
  }

  if (requested_size > 0 && !use_grow_oom &&
      releasable_bytes < required_releasable_bytes) {
    // Without a structured grow snapshot, use the current driver capacity to
    // decide whether remap plus a grow can satisfy the complete request.
    const auto driver_topup_start = Clock::now();
    compact_context.driver_memory = CollectDriverMemoryStats(place_.device);
    compact_context.driver_memory_collected = true;
    compact_context.driver_topup_us =
        ElapsedMicros(driver_topup_start, Clock::now());
    size_t driver_topup_bytes = required_releasable_bytes - releasable_bytes;
    const bool driver_can_top_up =
        compact_context.driver_memory.mem_info_status == phi::gpuSuccess &&
        compact_context.driver_memory.driver_actual_avail >= driver_topup_bytes;
    if (!driver_can_top_up) {
      LogCompactSkip(compact_state,
                     compact_context,
                     "insufficient_releasable_mapped_free");
      if (attempt_result != nullptr) {
        const size_t driver_available =
            compact_context.driver_memory.mem_info_status == phi::gpuSuccess
                ? compact_context.driver_memory.driver_actual_avail
                : 0;
        const size_t required_after_driver =
            required_releasable_bytes > driver_available
                ? required_releasable_bytes - driver_available
                : 0;
        attempt_result->status =
            VMMRemapAttemptStatus::kInsufficientMovableMemory;
        attempt_result->required_bytes = AlignedSize(
            required_after_driver, underlying_allocator_->handle_size());
      }
      return 0;
    }
    remap_target = releasable_bytes;
  }

  if (releasable_handles == 0) {
    LogCompactSkip(compact_state, compact_context, "no_releasable_handles");
    if (attempt_result != nullptr) {
      attempt_result->status = VMMRemapAttemptStatus::kNoMovableMemory;
    }
    return 0;
  }

  auto destination_policy =
      RemapTransaction::DestinationPolicy::kTailThenAnyGap;
  if (requested_size > 0) {
    const size_t handle_size = underlying_allocator_->handle_size();
    const size_t destination_bytes = AlignedSize(remap_target, handle_size);
    const size_t request_backing_bytes =
        AlignedSize(requested_size, handle_size);
    destination_policy =
        destination_bytes >= request_backing_bytes
            ? RemapTransaction::DestinationPolicy::kDirectGapThenTail
            : RemapTransaction::DestinationPolicy::kTailOnly;
  }

  const size_t driver_topup_bytes =
      required_releasable_bytes > releasable_bytes
          ? required_releasable_bytes - releasable_bytes
          : 0;
  VLOG(3) << "VMM V2 compact attempt: seq=" << compact_seq
          << " pool=" << static_cast<int>(pool_type_)
          << " requested=" << requested_size
          << " compact_target=" << compact_target
          << " remap_target=" << remap_target
          << " tail_free=" << compact_state.tail_free
          << " ready_bytes=" << releasable_bytes << " driver_actual_avail="
          << (compact_context.driver_memory_collected
                  ? compact_context.driver_memory.driver_actual_avail
                  : 0UL)
          << " driver_topup_bytes=" << driver_topup_bytes
          << " source_pages=" << source_pages.size();

  auto commit_destination_allocations =
      [this](std::vector<DecoratedAllocationPtr>* allocations) {
        TrackUnderlyingAllocationsOrRestore(allocations);
      };
  auto can_prepare_destination_range = [this](void* ptr, size_t size) {
    return CanPrepareDestinationRange(ptr, size);
  };
  auto prepare_destination_range = [this](void* ptr, size_t size) {
    return PrepareDestinationRange(ptr, size);
  };
  FreeBlockRemapCompactor compactor(underlying_allocator_,
                                    pool_type_,
                                    commit_destination_allocations,
                                    can_prepare_destination_range,
                                    prepare_destination_range);
  if (attempt_result != nullptr) {
    attempt_result->status = VMMRemapAttemptStatus::kAttempted;
  }
  const auto compactor_start = Clock::now();
  size_t remapped = 0;
  try {
    remapped = compactor.Compact(
        &all_blocks_, remap_target, source_pages, destination_policy);
  } catch (...) {
    // Remap can replace list nodes before ownership commit. Rebuild both
    // indexes after transaction rollback before propagating the failure.
    RebuildFreeBlockIndex();
    throw;
  }
  const uint64_t compactor_us = ElapsedMicros(compactor_start, Clock::now());
  // Source movement may replace FREE blocks with UNMAPPED-FREE/FREE segments
  // before destination setup fails. Rebuild to avoid stale index iterators.
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
    return false;
  }

  std::vector<BlockPart> collected;
  auto collect_ipc_parts = [&] {
    auto* output = parts != nullptr ? &collected : nullptr;
    return mark_ipc_exported
               ? underlying_allocator_->ExportIPCParts(target_va, size, output)
               : underlying_allocator_->CollectIPCParts(
                     target_va, size, output);
  };
  if (!collect_ipc_parts()) {
    const size_t cleared =
        underlying_allocator_->ClearRemapDestinationOwnershipInRange(target_va,
                                                                     size);
    if (cleared > 0) {
      collected.clear();
    }
    if (cleared == 0 || !collect_ipc_parts()) {
      VLOG(4) << "VMM V2 IPC backing lookup failed for " << ptr
              << " size=" << size;
      return false;
    }
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

std::vector<VMMAutoGrowthBestFitAllocatorV2::FreeBlockInfo>
VMMAutoGrowthBestFitAllocatorV2::SnapshotFreeBlockInfo() const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<FreeBlockInfo> info;
  info.reserve(free_blocks_.size());
  for (const auto& entry : free_blocks_) {
    info.emplace_back(entry.first.first,
                      reinterpret_cast<uintptr_t>(entry.first.second));
  }
  return info;
}

std::vector<VMMAutoGrowthBestFitAllocatorV2::BlockInfo>
VMMAutoGrowthBestFitAllocatorV2::SnapshotBlockInfo() const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<BlockInfo> info;
  info.reserve(all_blocks_.size());
  for (const auto& block : all_blocks_) {
    if (block.IsUnmappedFree()) {
      continue;
    }
    info.emplace_back(block.size(),
                      reinterpret_cast<uintptr_t>(block.ptr()),
                      !block.IsActive());
  }
  return info;
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
    mapped_remain.ClearRemapSafety();
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
    DecoratedAllocationPtr* allocation) {
  underlying_allocations_.Add(allocation);
}

void VMMAutoGrowthBestFitAllocatorV2::TrackUnderlyingAllocationsOrRestore(
    std::vector<DecoratedAllocationPtr>* allocations) {
  underlying_allocations_.AddAllOrRestore(allocations);
}

bool VMMAutoGrowthBestFitAllocatorV2::IsRemapDestinationAllocation(
    const DecoratedAllocationPtr& allocation) const {
  if (!underlying_allocator_->IsRemapDestinationAllocation(allocation->ptr())) {
    return false;
  }
  return true;
}

bool VMMAutoGrowthBestFitAllocatorV2::CanPrepareDestinationRange(
    void* ptr, size_t size) const {
  if (underlying_allocations_.AllOverlapsSatisfy(
          ptr, size, [this](const DecoratedAllocationPtr& allocation) {
            return IsRemapDestinationAllocation(allocation);
          })) {
    return true;
  }
  return underlying_allocator_->IsRangeUnmapped(
      reinterpret_cast<VMMDevicePtr>(ptr), size);
}

bool VMMAutoGrowthBestFitAllocatorV2::PrepareDestinationRange(void* ptr,
                                                              size_t size) {
  const bool released_owned_overlaps = underlying_allocations_.EraseOverlapsIf(
      ptr, size, [this](const DecoratedAllocationPtr& allocation) {
        if (!IsRemapDestinationAllocation(allocation)) {
          return false;
        }
        VLOG(4) << "VMM V2 remap destination preparation: releasing stale "
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
  TrackUnderlyingAllocation(&allocation);
  return block;
}

bool VMMAutoGrowthBestFitAllocatorV2::CanReleaseUnderlyingAllocation(
    uint8_t* base, size_t size) const {
  if (!IsRangeEntirelyFree(base, size)) {
    return false;
  }
  return underlying_allocator_->IsRangeReleasable(
      reinterpret_cast<VMMDevicePtr>(base), size);
}

bool VMMAutoGrowthBestFitAllocatorV2::HasReleasableUnderlyingAllocation(
    const UnderlyingRanges& entirely_free_ranges) const {
  for (const auto& range : entirely_free_ranges) {
    if (underlying_allocator_->IsRangeReleasable(range.first, range.second)) {
      return true;
    }
  }
  return false;
}

bool VMMAutoGrowthBestFitAllocatorV2::TryReleaseUnderlyingAllocation(
    UnderlyingAllocationRegistry::iterator* alloc_it,
    uint64_t* released,
    bool range_verified_free,
    BlockListIt* block_search_begin) {
  auto* allocation = (**alloc_it).get();
  auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
  const size_t alloc_size = allocation->size();
  const bool releasable =
      range_verified_free
          ? underlying_allocator_->IsRangeReleasable(
                reinterpret_cast<VMMDevicePtr>(base), alloc_size)
          : CanReleaseUnderlyingAllocation(base, alloc_size);
  if (!releasable) {
    return false;
  }

  if (underlying_allocator_->IsRemapDestinationAllocation(base)) {
    underlying_allocator_->ClearRemapDestinationOwnershipInRange(
        reinterpret_cast<VMMDevicePtr>(base), alloc_size);
  }
  if (block_search_begin == nullptr) {
    ReplaceRangeWithUnmappedFree(base, alloc_size);
  } else {
    *block_search_begin =
        ReplaceRangeWithUnmappedFree(base, alloc_size, *block_search_begin);
  }
  *released += alloc_size;
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

BlockListIt VMMAutoGrowthBestFitAllocatorV2::TryMergeUnmappedFree(
    BlockListIt it) {
  if (it == all_blocks_.end() || !it->IsUnmappedFree()) {
    return all_blocks_.end();
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
  return it;
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
  const auto release_start = Clock::now();
  std::lock_guard<SpinLock> guard(spinlock_);
  ReleaseTiming timing;
  timing.lock_wait_us = ElapsedMicros(release_start, Clock::now());
  const bool log_release_stats = VLOG_IS_ON(3);
  ReleaseStats before;
  auto op_start = Clock::now();
  if (log_release_stats) {
    before = CollectReleaseStats();
  }
  const auto entirely_free_ranges = CollectEntirelyFreeUnderlyingRanges();
  const auto partial_release_plans = CollectPartialReleasePlans();
  timing.precheck_us = ElapsedMicros(op_start, Clock::now());
  const bool has_releasable =
      (log_release_stats
           ? before.releasable_backing_count > 0
           : HasReleasableUnderlyingAllocation(entirely_free_ranges)) ||
      !partial_release_plans.empty();
  if (!has_releasable) {
    if (log_release_stats) {
      timing.total_us = ElapsedMicros(release_start, Clock::now());
      LogReleaseStats(before,
                      before,
                      /*released_bytes=*/0,
                      timing,
                      CUDAVirtualMemAllocatorV2::ReleaseDriverStats{});
    }
    return 0;
  }
  // FreeIdleChunks may release CUDA VMM mappings and physical handles. Those
  // driver calls are not ordered by the stream-safe wrapper, so wait before
  // making any previously returned VA range invalid.
  platform::CUDADeviceGuard device_guard(place_.device);
  op_start = Clock::now();
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  timing.device_sync_us = ElapsedMicros(op_start, Clock::now());
  const auto driver_before = underlying_allocator_->GetReleaseDriverStats();
  op_start = Clock::now();
  const uint64_t released =
      FreeIdleChunks(entirely_free_ranges, partial_release_plans);
  timing.release_us = ElapsedMicros(op_start, Clock::now());
  if (log_release_stats) {
    op_start = Clock::now();
    const auto after = CollectReleaseStats();
    timing.post_stats_us = ElapsedMicros(op_start, Clock::now());
    timing.total_us = ElapsedMicros(release_start, Clock::now());
    const auto driver_after = underlying_allocator_->GetReleaseDriverStats();
    CUDAVirtualMemAllocatorV2::ReleaseDriverStats driver_delta;
    driver_delta.allocation_count =
        driver_after.allocation_count - driver_before.allocation_count;
    driver_delta.handle_count =
        driver_after.handle_count - driver_before.handle_count;
    driver_delta.released_bytes =
        driver_after.released_bytes - driver_before.released_bytes;
    driver_delta.skipped_owned_handles = driver_after.skipped_owned_handles -
                                         driver_before.skipped_owned_handles;
    driver_delta.unmap_calls =
        driver_after.unmap_calls - driver_before.unmap_calls;
    driver_delta.unmap_us = driver_after.unmap_us - driver_before.unmap_us;
    driver_delta.release_calls =
        driver_after.release_calls - driver_before.release_calls;
    driver_delta.release_us =
        driver_after.release_us - driver_before.release_us;
    driver_delta.metadata_us =
        driver_after.metadata_us - driver_before.metadata_us;
    LogReleaseStats(before, after, released, timing, driver_delta);
  }
  return released;
}

uint64_t VMMAutoGrowthBestFitAllocatorV2::FreeIdleChunks(
    const UnderlyingRanges& entirely_free_ranges,
    const PartialReleasePlans& partial_release_plans) {
  uint64_t released = 0;
  auto block_search_begin = all_blocks_.begin();
  for (const auto& range : entirely_free_ranges) {
    auto alloc_it = underlying_allocations_.FindByAddress(range.first);
    if (alloc_it == underlying_allocations_.end() ||
        (*alloc_it)->size() != range.second) {
      continue;
    }
    TryReleaseUnderlyingAllocation(&alloc_it,
                                   &released,
                                   /*range_verified_free=*/true,
                                   &block_search_begin);
  }
  block_search_begin = all_blocks_.begin();
  released += ReleasePartialBacking(partial_release_plans, &block_search_begin);

  TrimTrailingUnmappedFreeBlocks();
  underlying_allocator_->SetTailOffset(ComputeTailOffset());
  return released;
}

VMMAutoGrowthBestFitAllocatorV2::UnderlyingRanges
VMMAutoGrowthBestFitAllocatorV2::CollectEntirelyFreeUnderlyingRanges() const {
  const auto backing_ranges = underlying_allocations_.CollectRangesByAddress();
  UnderlyingRanges entirely_free_ranges;
  entirely_free_ranges.reserve(backing_ranges.size());
  auto first_block = all_blocks_.begin();
  for (const auto& range : backing_ranges) {
    const auto range_end = range.first + range.second;
    while (first_block != all_blocks_.end() &&
           first_block->end_va() <= range.first) {
      ++first_block;
    }
    bool contains_active_block = false;
    for (auto block_it = first_block;
         block_it != all_blocks_.end() && block_it->begin_va() < range_end;
         ++block_it) {
      if (block_it->IsActive()) {
        contains_active_block = true;
        break;
      }
    }
    if (!contains_active_block) {
      entirely_free_ranges.push_back(range);
    }
  }
  return entirely_free_ranges;
}

VMMAutoGrowthBestFitAllocatorV2::PartialReleasePlans
VMMAutoGrowthBestFitAllocatorV2::CollectPartialReleasePlans() const {
  PartialReleasePlans plans;
  const size_t handle_size = underlying_allocator_->handle_size();
  if (handle_size == 0) {
    return plans;
  }

  const VMMDevicePtr virtual_base = underlying_allocator_->virtual_mem_base();
  const auto backing_ranges = underlying_allocations_.CollectRangesByAddress();
  auto first_block = all_blocks_.begin();
  for (const auto& backing : backing_ranges) {
    const VMMDevicePtr backing_end = backing.first + backing.second;
    while (first_block != all_blocks_.end() &&
           first_block->end_va() <= backing.first) {
      ++first_block;
    }

    PartialReleasePlan plan;
    plan.allocation_base = backing.first;
    auto append_range = [&plan](VMMDevicePtr va, size_t size) {
      if (!plan.ranges.empty() &&
          plan.ranges.back().first + plan.ranges.back().second == va) {
        plan.ranges.back().second += size;
      } else {
        plan.ranges.emplace_back(va, size);
      }
    };

    for (auto block_it = first_block;
         block_it != all_blocks_.end() && block_it->begin_va() < backing_end;
         ++block_it) {
      if (!block_it->IsMappedFree()) {
        continue;
      }
      const VMMDevicePtr overlap_begin =
          std::max(block_it->begin_va(), backing.first);
      const VMMDevicePtr overlap_end =
          std::min(block_it->end_va(), backing_end);
      const size_t begin_offset = overlap_begin - virtual_base;
      const size_t end_offset = overlap_end - virtual_base;
      const size_t first_handle =
          begin_offset / handle_size +
          static_cast<size_t>(begin_offset % handle_size != 0);
      const size_t end_handle = end_offset / handle_size;
      if (first_handle >= end_handle) {
        continue;
      }

      const VMMDevicePtr candidate_begin =
          virtual_base + first_handle * handle_size;
      const size_t candidate_size = (end_handle - first_handle) * handle_size;
      if (underlying_allocator_->IsRangeReleasable(candidate_begin,
                                                   candidate_size)) {
        append_range(candidate_begin, candidate_size);
        continue;
      }
      for (VMMDevicePtr va = candidate_begin;
           va < candidate_begin + candidate_size;
           va += handle_size) {
        if (underlying_allocator_->IsRangeReleasable(va, handle_size)) {
          append_range(va, handle_size);
        }
      }
    }
    if (!plan.ranges.empty()) {
      plans.push_back(std::move(plan));
    }
  }
  return plans;
}

uint64_t VMMAutoGrowthBestFitAllocatorV2::ReleasePartialBacking(
    const PartialReleasePlans& plans, BlockListIt* block_search_begin) {
  uint64_t released = 0;
  for (const auto& plan : plans) {
    auto alloc_it = underlying_allocations_.FindByAddress(plan.allocation_base);
    if (alloc_it == underlying_allocations_.end()) {
      continue;
    }

    auto allocation = underlying_allocations_.Take(alloc_it);
    CUDAVirtualMemAllocatorV2::PartialReleaseResult result;
    try {
      result = underlying_allocator_->ReleaseFreeHandleRanges(
          &allocation,
          plan.ranges,
          [this](std::vector<DecoratedAllocationPtr>* remaining_allocations) {
            TrackUnderlyingAllocationsOrRestore(remaining_allocations);
          });
    } catch (...) {
      if (allocation != nullptr) {
        underlying_allocations_.Add(&allocation);
      }
      throw;
    }
    if (allocation != nullptr) {
      underlying_allocations_.Add(&allocation);
      continue;
    }

    for (const auto& range : result.released_ranges) {
      *block_search_begin =
          ReplaceRangeWithUnmappedFree(reinterpret_cast<uint8_t*>(range.first),
                                       range.second,
                                       *block_search_begin);
    }
    released += result.released_bytes;
  }
  return released;
}

VMMAutoGrowthBestFitAllocatorV2::ReleaseStats
VMMAutoGrowthBestFitAllocatorV2::CollectReleaseStats() const {
  struct BackingView {
    uint8_t* base{nullptr};
    uint8_t* end{nullptr};
    size_t size{0};
    size_t active_bytes{0};
    size_t mapped_free_bytes{0};
  };

  ReleaseStats stats;
  std::vector<BackingView> backings;
  backings.reserve(static_cast<size_t>(std::distance(
      underlying_allocations_.begin(), underlying_allocations_.end())));
  for (const auto& allocation : underlying_allocations_) {
    auto* base = reinterpret_cast<uint8_t*>(allocation->ptr());
    backings.push_back(
        {base, base + allocation->size(), allocation->size(), 0, 0});
  }
  std::sort(backings.begin(),
            backings.end(),
            [](const BackingView& lhs, const BackingView& rhs) {
              return lhs.base < rhs.base;
            });

  for (const auto& block : all_blocks_) {
    if (block.IsActive()) {
      stats.active_bytes += block.size_;
    } else if (block.IsMappedFree()) {
      stats.mapped_free_bytes += block.size_;
    } else if (block.IsUnmappedFree()) {
      stats.unmapped_free_bytes += block.size_;
    }
  }

  auto first_block = all_blocks_.begin();
  for (auto& backing : backings) {
    while (first_block != all_blocks_.end() &&
           first_block->end_ptr() <= backing.base) {
      ++first_block;
    }
    for (auto block_it = first_block;
         block_it != all_blocks_.end() && block_it->begin_ptr() < backing.end;
         ++block_it) {
      const size_t overlap = RangeOverlapBytes(
          backing.base, backing.size, block_it->ptr_, block_it->size_);
      if (block_it->IsActive()) {
        backing.active_bytes += overlap;
      } else if (block_it->IsMappedFree()) {
        backing.mapped_free_bytes += overlap;
      }
    }

    ++stats.backing_count;
    stats.backing_bytes += backing.size;
    const bool entirely_free = backing.active_bytes == 0;
    const bool releasable =
        entirely_free &&
        underlying_allocator_->IsRangeReleasable(
            reinterpret_cast<VMMDevicePtr>(backing.base), backing.size);
    if (releasable) {
      ++stats.releasable_backing_count;
      stats.releasable_backing_bytes += backing.size;
    } else {
      stats.stranded_mapped_free_bytes += backing.mapped_free_bytes;
      if (entirely_free) {
        ++stats.release_blocked_backing_count;
        stats.release_blocked_backing_bytes += backing.size;
      }
    }
    if (backing.active_bytes > 0 && backing.mapped_free_bytes > 0) {
      ++stats.mixed_backing_count;
      stats.mixed_backing_bytes += backing.size;
    }
  }

  size_t first_backing = 0;
  for (const auto& block : all_blocks_) {
    if (!block.IsActive()) {
      continue;
    }
    while (first_backing < backings.size() &&
           backings[first_backing].end <= block.begin_ptr()) {
      ++first_backing;
    }
    size_t overlap_count = 0;
    for (size_t i = first_backing;
         i < backings.size() && backings[i].base < block.end_ptr();
         ++i) {
      if (backings[i].end > block.begin_ptr()) {
        ++overlap_count;
      }
    }
    if (overlap_count > 1) {
      ++stats.active_blocks_crossing_backings;
      stats.active_bytes_crossing_backings += block.size_;
    }
  }
  return stats;
}

void VMMAutoGrowthBestFitAllocatorV2::LogReleaseStats(
    const ReleaseStats& before,
    const ReleaseStats& after,
    uint64_t released_bytes,
    const ReleaseTiming& timing,
    const CUDAVirtualMemAllocatorV2::ReleaseDriverStats& driver_stats) const {
  if (before.backing_count == 0 && after.backing_count == 0) {
    return;
  }
  VLOG(3) << "Allocator backing release: allocator=vmm_v2"
          << " device=" << static_cast<int>(place_.device)
          << " pool=" << (pool_type_ == PoolType::kSmall ? "small" : "large")
          << " before_backings=" << before.backing_count
          << " before_backing_bytes=" << before.backing_bytes
          << " before_releasable_backings=" << before.releasable_backing_count
          << " before_releasable_bytes=" << before.releasable_backing_bytes
          << " before_release_blocked_backings="
          << before.release_blocked_backing_count
          << " before_release_blocked_bytes="
          << before.release_blocked_backing_bytes
          << " before_mixed_backings=" << before.mixed_backing_count
          << " before_mixed_backing_bytes=" << before.mixed_backing_bytes
          << " before_active_bytes=" << before.active_bytes
          << " before_mapped_free_bytes=" << before.mapped_free_bytes
          << " before_stranded_mapped_free_bytes="
          << before.stranded_mapped_free_bytes
          << " before_unmapped_free_bytes=" << before.unmapped_free_bytes
          << " before_cross_backing_active_blocks="
          << before.active_blocks_crossing_backings
          << " before_cross_backing_active_bytes="
          << before.active_bytes_crossing_backings
          << " release_operations=" << driver_stats.allocation_count
          << " released_bytes=" << released_bytes
          << " after_backings=" << after.backing_count
          << " after_backing_bytes=" << after.backing_bytes
          << " after_mapped_free_bytes=" << after.mapped_free_bytes
          << " after_stranded_mapped_free_bytes="
          << after.stranded_mapped_free_bytes
          << " after_unmapped_free_bytes=" << after.unmapped_free_bytes
          << " lock_wait_us=" << timing.lock_wait_us
          << " precheck_us=" << timing.precheck_us
          << " device_sync_us=" << timing.device_sync_us
          << " release_us=" << timing.release_us
          << " post_stats_us=" << timing.post_stats_us
          << " total_us=" << timing.total_us
          << " driver_allocations=" << driver_stats.allocation_count
          << " driver_handles=" << driver_stats.handle_count
          << " driver_released_bytes=" << driver_stats.released_bytes
          << " driver_skipped_owned_handles="
          << driver_stats.skipped_owned_handles
          << " driver_unmap_calls=" << driver_stats.unmap_calls
          << " driver_unmap_us=" << driver_stats.unmap_us
          << " driver_release_calls=" << driver_stats.release_calls
          << " driver_release_us=" << driver_stats.release_us
          << " driver_metadata_us=" << driver_stats.metadata_us;
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
  // blocks in the overlapping VA range before the destination allocation
  // is processed).  FreeImpl handles this safely: original allocation
  // skips remapped handles; the destination allocation unmaps and releases its
  // own handles.
  return true;
}

void VMMAutoGrowthBestFitAllocatorV2::ReplaceRangeWithUnmappedFree(
    uint8_t* base, size_t size) {
  (void)ReplaceRangeWithUnmappedFree(base, size, all_blocks_.begin());
}

BlockListIt VMMAutoGrowthBestFitAllocatorV2::ReplaceRangeWithUnmappedFree(
    uint8_t* base, size_t size, BlockListIt search_begin) {
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

  auto it = search_begin;
  auto insert_pos = all_blocks_.end();
  while (it != all_blocks_.end()) {
    auto* bptr = it->begin_ptr();
    auto* bend = it->end_ptr();

    if (bend <= base) {
      ++it;
      continue;
    }
    if (bptr >= end) {
      insert_pos = it;
      break;
    }

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
      insert_pos = it;
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

      erase_free_index(it);
      it->TrimToPrefix(left_size);
      insert_free_index(it);
      auto right_it = all_blocks_.insert(std::next(it), std::move(right));
      insert_free_index(right_it);
      insert_pos = right_it;
      break;  // done
    }

    ++it;
  }

  if (insert_pos == all_blocks_.end()) {
    insert_pos = it;
  }
  auto unmapped_it = all_blocks_.insert(
      insert_pos, BlockV2::MakeUnmappedFreeBlock(base, size, pool_type_));
  InsertUnmappedFreeBlock(unmapped_it);
  return TryMergeUnmappedFree(unmapped_it);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
