// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <mutex>
#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/core/memory/allocation/aligned_allocator.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator.h"

PHI_DEFINE_EXPORTED_uint64(
    vmm_small_pool_size_in_mb,
    1,
    "Threshold (MiB) separating the small and large pools. "
    "0 disables the small pool and enables single-pool mode "
    "(all requests go to the large pool). When > 0, requests "
    "<= threshold use the small pool; larger requests use the "
    "large pool. Default: 0.");
PHI_DEFINE_EXPORTED_uint64(vmm_small_pool_min_growth_size_in_mb,
                           0,
                           "The minimal chunk size for the small pool in MiB. "
                           "If small_pool_size_in_mb > 0, this overrides "
                           "the constructor-provided global growth size "
                           "(FLAGS_auto_growth_chunk_size_in_mb).");
PHI_DEFINE_EXPORTED_uint64(vmm_large_pool_min_growth_size_in_mb,
                           0,
                           "The minimal chunk size for the large pool in MiB. "
                           "If small_pool_size_in_mb > 0, this overrides "
                           "the constructor-provided global growth size "
                           "(FLAGS_auto_growth_chunk_size_in_mb).");
PHI_DEFINE_EXPORTED_uint64(
    vmm_large_pool_pre_alloc_in_mb,
    0,
    "Pre-reserve this many MiB in the large pool. 0 disables pre-allocation.");
PHI_DEFINE_EXPORTED_uint64(
    vmm_small_pool_pre_alloc_in_mb,
    0,
    "Pre-reserve this many MiB in the small pool. 0 disables pre-allocation.");
PHI_DEFINE_EXPORTED_uint64(
    vmm_pre_alloc_in_mb,
    0,
    "Pre-reserve this many MiB in the small pool. 0 disables pre-allocation.");
PHI_DEFINE_EXPORTED_bool(
    dump_vmm_allocation_info,
    false,
    "dump VirtualMemoryAutoGrowthBestFitAllocator's allocation info");
PHI_DEFINE_EXPORTED_bool(native_compact,
                         false,
                         "native_compact means compact memory after OOM, The "
                         "algorithm still needs to be upgraded.");

namespace paddle {
namespace memory {
namespace allocation {

bool NeedSplit(size_t block_size, size_t alignment, size_t alloc_size) {
  return block_size > (alloc_size * 2) || (block_size - alloc_size) > alignment;
}

// Merge if two parts refer to the same chunk and touch each other.
static inline bool TryConcatAdjacent(BlockPart *a, const BlockPart &b) {
  if (!a) return false;
  if (a->chunk.get() != b.chunk.get()) return false;
  if (a->chunk_rel_off + a->len != b.chunk_rel_off) return false;
  a->len += b.len;
  return true;
}

static std::vector<BlockPart> SlicePartsForRange(
    const std::vector<BlockPart> &parts, size_t pick_off, size_t pick_len) {
  std::vector<BlockPart> out;
  if (pick_len == 0 || parts.empty()) {
    return out;
  }

  PADDLE_ENFORCE_LE(
      pick_off,
      std::numeric_limits<size_t>::max() - pick_len,
      common::errors::InvalidArgument(
          "Invalid VMM block-part slice range: offset %zu plus length %zu "
          "overflows.",
          pick_off,
          pick_len));

  if (parts.size() == 1) {
    const auto &p = parts.front();
    PADDLE_ENFORCE_LE(
        pick_off,
        p.len,
        common::errors::InvalidArgument(
            "Invalid VMM block-part slice offset %zu for part length %zu.",
            pick_off,
            p.len));
    PADDLE_ENFORCE_LE(
        pick_len,
        p.len - pick_off,
        common::errors::InvalidArgument(
            "Invalid VMM block-part slice length %zu at offset %zu for part "
            "length %zu.",
            pick_len,
            pick_off,
            p.len));
    return {BlockPart{p.chunk, p.chunk_rel_off + pick_off, pick_len}};
  }

  out.reserve(parts.size());
  const size_t pick_end = pick_off + pick_len;
  size_t cursor = 0;
  size_t sliced_len = 0;
  for (const auto &p : parts) {
    const size_t part_begin = cursor;
    const size_t part_end = cursor + p.len;
    cursor = part_end;

    if (part_end <= pick_off) {
      continue;
    }
    if (part_begin >= pick_end) {
      break;
    }

    const size_t slice_begin = std::max(part_begin, pick_off);
    const size_t slice_end = std::min(part_end, pick_end);
    BlockPart cut{p.chunk,
                  p.chunk_rel_off + (slice_begin - part_begin),
                  slice_end - slice_begin};
    if (!out.empty() && TryConcatAdjacent(&out.back(), cut)) {
      sliced_len += cut.len;
      continue;
    }
    out.push_back(std::move(cut));
    sliced_len += out.back().len;
  }
  PADDLE_ENFORCE_EQ(
      sliced_len,
      pick_len,
      common::errors::InvalidArgument(
          "Invalid VMM block-part slice range: requested %zu bytes at offset "
          "%zu, but only sliced %zu bytes from %zu parts.",
          pick_len,
          pick_off,
          sliced_len,
          parts.size()));
  return out;
}

static inline void AppendPartsTail(std::vector<BlockPart> *dst,
                                   std::vector<BlockPart> *src) {
  if (src->empty()) return;
  dst->reserve(dst->size() + src->size());
  auto begin = src->begin();
  if (!dst->empty() && TryConcatAdjacent(&dst->back(), src->front())) {
    ++begin;
  }
  dst->insert(dst->end(),
              std::make_move_iterator(begin),
              std::make_move_iterator(src->end()));
}

static inline bool ShouldLogAllocatorStats(uint64_t seq) {
  return seq <= 10 || seq % 10000 == 0;
}

VirtualMemoryAutoGrowthBestFitAllocator::
    VirtualMemoryAutoGrowthBestFitAllocator(
        const std::shared_ptr<Allocator> &underlying_allocator,
        size_t alignment,
        const GPUPlace &place)
    : underlying_allocator_(
          std::make_shared<AlignedAllocator>(underlying_allocator, alignment)),
      alignment_(alignment),
      place_(place) {
  // NOTE(liujinnan): Only support TotalMemoryCompactor strategy for now.
  memory_compactor_ = std::make_unique<TotalMemoryCompactor>();
}

void VirtualMemoryAutoGrowthBestFitAllocator::MaybeLogAllocatorStats(
    const char *reason, uint64_t seq) const {
  if (!VLOG_IS_ON(4) || !ShouldLogAllocatorStats(seq)) {
    return;
  }
  VLOG(4) << "[VMM][AllocatorStats]"
          << " reason=" << reason
          << " alloc_from_free_calls=" << alloc_from_free_calls_
          << " hits=" << alloc_from_free_hits_
          << " misses=" << alloc_from_free_misses_ << " splits=" << split_count_
          << " no_splits=" << no_split_count_
          << " exact_fit=" << exact_fit_count_
          << " near_fit_no_split=" << near_fit_no_split_count_
          << " single_part_splits=" << single_part_split_count_
          << " multi_part_splits=" << multi_part_split_count_
          << " free_calls=" << free_calls_
          << " merge_prev=" << free_merge_prev_count_
          << " merge_next=" << free_merge_next_count_
          << " merge_both=" << free_merge_both_count_
          << " merge_none=" << free_merge_none_count_
          << " free_blocks=" << free_blocks_.size()
          << " all_blocks=" << all_blocks_.size();
}

phi::Allocation *VirtualMemoryAutoGrowthBestFitAllocator::AllocateImpl(
    size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size = AlignedSize(size, alignment_);
  auto result = AllocFromFreeBlocks(size);

  if (!result) {
    ExtendOrCompact(size);
    result = AllocFromFreeBlocks(size);
  }

  return result;
}

void VirtualMemoryAutoGrowthBestFitAllocator::FreeImpl(
    phi::Allocation *allocation) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto block_it = static_cast<BlockAllocation *>(allocation)->block_it_;
  TryMergeBlock2Blocks(block_it);
  delete allocation;
}

bool VirtualMemoryAutoGrowthBestFitAllocator::CollectTensorParts(
    void *ptr, size_t size, std::vector<BlockPart> *parts) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto target_begin = reinterpret_cast<uintptr_t>(ptr);
  PADDLE_ENFORCE_LE(
      size,
      std::numeric_limits<uintptr_t>::max() - target_begin,
      common::errors::InvalidArgument(
          "Invalid VMM tensor range: ptr %p plus size %zu overflows.",
          ptr,
          size));
  auto target_end = target_begin + size;
  for (const auto &block : all_blocks_) {
    if (block.is_free_) {
      continue;
    }
    auto block_begin = reinterpret_cast<uintptr_t>(block.ptr_);
    auto block_end = block_begin + block.size_;
    if (target_begin >= block_begin && target_end <= block_end) {
      if (parts) {
        *parts =
            SlicePartsForRange(block.parts_, target_begin - block_begin, size);
      }
      return true;
    }
  }
  return false;
}

void VirtualMemoryAutoGrowthBestFitAllocator::TryMergeBlock2Blocks(
    std::list<Block>::iterator block) {
  if (VLOG_IS_ON(4)) {
    const bool can_merge_prev =
        block != all_blocks_.begin() && std::prev(block)->is_free_ &&
        reinterpret_cast<uint8_t *>(std::prev(block)->ptr_) +
                std::prev(block)->size_ ==
            block->ptr_;
    const bool can_merge_next =
        block != std::prev(all_blocks_.end()) && std::next(block)->is_free_ &&
        reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ ==
            std::next(block)->ptr_;
    ++free_calls_;
    if (can_merge_prev && can_merge_next) {
      ++free_merge_both_count_;
    } else if (can_merge_prev) {
      ++free_merge_prev_count_;
    } else if (can_merge_next) {
      ++free_merge_next_count_;
    } else {
      ++free_merge_none_count_;
    }
    MaybeLogAllocatorStats("free", free_calls_);
  }

  if (block->ptr_ == all_blocks_.front().ptr_ &&
      block->ptr_ == all_blocks_.back().ptr_) {
    block->is_free_ = true;
    free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
  } else if (block->ptr_ == all_blocks_.front().ptr_) {
    auto next = std::next(block);
    if (next->is_free_ &&
        reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ == next->ptr_) {
      // merge with next
      AppendPartsTail(&block->parts_, &next->parts_);
      block->size_ += next->size_;
      block->is_free_ = true;
      free_blocks_.erase(std::make_pair(next->size_, next->ptr_));
      all_blocks_.erase(next);
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    } else {
      block->is_free_ = true;
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    }
  } else if (block->ptr_ == all_blocks_.back().ptr_) {
    auto pre = std::prev(block);
    if (pre->is_free_ &&
        reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ == block->ptr_) {
      // merge with pre
      free_blocks_.erase(std::make_pair(pre->size_, pre->ptr_));
      AppendPartsTail(&pre->parts_, &block->parts_);
      pre->size_ += block->size_;
      all_blocks_.erase(block);
      free_blocks_.emplace(std::make_pair(pre->size_, pre->ptr_), pre);
    } else {
      block->is_free_ = true;
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    }
  } else {
    auto pre = std::prev(block);
    auto next = std::next(block);
    if (pre->is_free_ &&
        reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ == block->ptr_ &&
        !(next->is_free_ &&
          reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ ==
              next->ptr_)) {
      // merge with pre
      free_blocks_.erase(std::make_pair(pre->size_, pre->ptr_));
      AppendPartsTail(&pre->parts_, &block->parts_);
      pre->size_ += block->size_;
      all_blocks_.erase(block);
      free_blocks_.emplace(std::make_pair(pre->size_, pre->ptr_), pre);
    } else if (next->is_free_ &&
               reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ ==
                   next->ptr_ &&
               !(pre->is_free_ &&
                 reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ ==
                     block->ptr_)) {
      // merge with next
      block->size_ += next->size_;
      block->is_free_ = true;
      AppendPartsTail(&block->parts_, &next->parts_);
      free_blocks_.erase(std::make_pair(next->size_, next->ptr_));
      all_blocks_.erase(next);
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    } else if (pre->is_free_ &&
               reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ ==
                   block->ptr_ &&
               next->is_free_ &&
               reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ ==
                   next->ptr_) {
      // merge with pre and next
      free_blocks_.erase(std::make_pair(pre->size_, pre->ptr_));
      free_blocks_.erase(std::make_pair(next->size_, next->ptr_));
      AppendPartsTail(&pre->parts_, &block->parts_);
      AppendPartsTail(&pre->parts_, &next->parts_);
      pre->size_ += (block->size_ + next->size_);
      all_blocks_.erase(block);
      all_blocks_.erase(next);
      free_blocks_.emplace(std::make_pair(pre->size_, pre->ptr_), pre);
    } else {
      block->is_free_ = true;
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    }
  }
}

std::optional<AllocationPtr>
VirtualMemoryAutoGrowthBestFitAllocator::AllocateOrCompact(size_t size) {
  AllocationPtr allocateptr = nullptr;
  // Just Allocate, no compact.
  if (!FLAGS_native_compact) {
    if (all_blocks_.empty()) {
      allocateptr = std::move(underlying_allocator_->Allocate(size));
    } else {
      auto free_block = std::prev(all_blocks_.end());
      if (free_block->is_free_) {
        assert(free_block->size_ < size);
        auto remain_size = size - free_block->size_;
        VLOG(4) << " Tail free block size {" << free_block->size_
                << "} is smaller than allocate size {" << size
                << "} after compact, re-alloc {" << remain_size << "}";
        allocateptr = std::move(underlying_allocator_->Allocate(remain_size));
      } else {
        VLOG(4) << "Tail block is not free, just allocate {" << size << "}";
        allocateptr = std::move(underlying_allocator_->Allocate(size));
      }
    }
    return allocateptr;
  }
  // Compact branch, try allocate and compact.
  try {
    allocateptr = std::move(underlying_allocator_->Allocate(size));
  } catch (const paddle::memory::allocation::BadAlloc &e) {
    VLOG(4) << "Do Memory Compact allocate size and compact " << size;
    size_t compact_free_size = memory_compactor_->Compact(
        all_blocks_, all_blocks_.front().ptr_, all_blocks_.back().ptr_);
    VLOG(4) << "Memory Compacted Size: " << compact_free_size;
    auto free_block = std::prev(all_blocks_.end());
    if (free_block->is_free_ && free_block->size_ < size) {
      auto realloc_size = size - free_block->size_;
      VLOG(4) << "Free block size {" << free_block->size_
              << "} is smaller than allocate size {" << size
              << "} after compact, re-alloc {" << realloc_size << "}";
      try {
        auto realloc_ptr =
            underlying_allocator_->Allocate(size - free_block->size_);
        VLOG(4) << "Re-alloc size {" << realloc_ptr->size() << "} success";
        free_block->size_ += realloc_ptr->size();
        allocations_.push_back(std::move(realloc_ptr));  // hold allocation
      } catch (const paddle::memory::allocation::BadAlloc &e) {
        VLOG(4) << "Re-alloc size {" << realloc_size << "} failed";
        throw;
      }
    }
    return std::nullopt;
  }
  return allocateptr;
}

void VirtualMemoryAutoGrowthBestFitAllocator::ExtendOrCompact(size_t size) {
  void *alloc_ptr = nullptr;
  size_t alloc_size = 0;
  if (FLAGS_dump_vmm_allocation_info) {
    DumpInfo("===== Before ExtendOrCompact ===== request size: " +
             std::to_string(size));
  }

  auto allocateptr = AllocateOrCompact(size).value_or(nullptr);
  if (!allocateptr) {
    // Allocate failed and Compact success branch.
    free_blocks_.clear();
    auto free_block = std::prev(all_blocks_.end());
    if (free_block->is_free_) {
      free_blocks_.emplace(std::make_pair(free_block->size_, free_block->ptr_),
                           free_block);
    } else {
      LOG(INFO) << "Dont have free block after memory compact";
    }
    if (FLAGS_dump_vmm_allocation_info) {
      DumpInfo("===== After ExtendOrCompact do compact =====");
    }
    // After compact, Merge is not needed. just return.
    return;
  }

  alloc_ptr = allocateptr->ptr();
  alloc_size = allocateptr->size();
  allocations_.push_back(std::move(allocateptr));  // hold allocation

  std::vector<BlockPart> new_parts;
  auto chunk = std::make_shared<VmmChunkMeta>();
  chunk->base = reinterpret_cast<VmmDevicePtr>(alloc_ptr);
  chunk->size = alloc_size;
#ifdef PADDLE_WITH_CUDA
  auto handle = CUDAVirtualMemAllocator::GetHandleFromBasePtr(alloc_ptr);
  PADDLE_ENFORCE_NE(
      handle,
      0,
      common::errors::InvalidArgument(
          "Allocation returned by underlying allocator is not VMM allocation"));
  chunk->handle = handle;
#else
  PADDLE_THROW(common::errors::Unavailable(
      "Virtual memory auto-growth allocator requires CUDA support."));
#endif
  chunk->device = place_.device;
  new_parts.emplace_back(BlockPart{chunk, 0, alloc_size});

  if (all_blocks_.empty()) {
    all_blocks_.emplace_back(alloc_ptr, alloc_size, true);
    auto it = all_blocks_.begin();
    it->parts_ = std::move(new_parts);
    free_blocks_.emplace(std::make_pair(alloc_size, alloc_ptr), it);
    return;
  }

  // insert to back
  auto block_it = all_blocks_.end();
  block_it--;
  if (block_it->is_free_ &&
      reinterpret_cast<uint8_t *>(block_it->ptr_) + block_it->size_ ==
          alloc_ptr) {
    // merge with pre
    free_blocks_.erase(std::make_pair(block_it->size_, block_it->ptr_));
    block_it->size_ += alloc_size;
    AppendPartsTail(&block_it->parts_, &new_parts);
    free_blocks_.emplace(std::make_pair(block_it->size_, block_it->ptr_),
                         block_it);
  } else {
    // do not merge
    all_blocks_.emplace_back(alloc_ptr, alloc_size, true);
    auto block_it = all_blocks_.end();
    block_it--;
    block_it->parts_ = std::move(new_parts);
    free_blocks_.emplace(std::make_pair(alloc_size, alloc_ptr), block_it);
  }
  if (FLAGS_dump_vmm_allocation_info) {
    DumpInfo("===== After ExtendOrCompact =====  request size: " +
             std::to_string(size) +
             " alloc size: " + std::to_string(alloc_size));
  }
}

phi::Allocation *VirtualMemoryAutoGrowthBestFitAllocator::AllocFromFreeBlocks(
    size_t size) {
  if (VLOG_IS_ON(4)) {
    ++alloc_from_free_calls_;
  }
  auto iter = free_blocks_.lower_bound(std::make_pair(size, nullptr));
  if (iter != free_blocks_.end()) {
    std::list<Block>::iterator block_it = iter->second;
    free_blocks_.erase(iter);
    if (VLOG_IS_ON(4)) {
      ++alloc_from_free_hits_;
    }
    if (NeedSplit(block_it->size_, alignment_, size)) {
      if (VLOG_IS_ON(4)) {
        ++split_count_;
        if (block_it->parts_.size() == 1) {
          ++single_part_split_count_;
        } else {
          ++multi_part_split_count_;
        }
        MaybeLogAllocatorStats("alloc_split", alloc_from_free_calls_);
      }
      void *remaining_ptr = reinterpret_cast<uint8_t *>(block_it->ptr_) + size;
      size_t remaining_size = block_it->size_ - size;

      std::vector<BlockPart> alloc_parts =
          SlicePartsForRange(block_it->parts_, 0, size);
      std::vector<BlockPart> remaining_parts =
          SlicePartsForRange(block_it->parts_, size, remaining_size);

      block_it->size_ = size;
      block_it->is_free_ = false;
      block_it->parts_.swap(alloc_parts);

      auto remaining_free_block = all_blocks_.insert(
          std::next(block_it), Block(remaining_ptr, remaining_size, true));
      remaining_free_block->parts_ = std::move(remaining_parts);
      free_blocks_.emplace(std::make_pair(remaining_size, remaining_ptr),
                           remaining_free_block);
    } else {
      if (VLOG_IS_ON(4)) {
        ++no_split_count_;
        if (block_it->size_ == size) {
          ++exact_fit_count_;
        } else {
          ++near_fit_no_split_count_;
        }
        MaybeLogAllocatorStats("alloc_no_split", alloc_from_free_calls_);
      }
      block_it->is_free_ = false;
    }
    return new BlockAllocation(block_it, place_);
  }
  if (VLOG_IS_ON(4)) {
    ++alloc_from_free_misses_;
    MaybeLogAllocatorStats("alloc_miss", alloc_from_free_calls_);
  }
  return nullptr;
}

size_t VirtualMemoryAutoGrowthBestFitAllocator::CompactImpl(
    const Place &place) {
  VLOG(1) << "Do Memory Compact Manual";
  size_t compact_free_size = memory_compactor_->Compact(
      all_blocks_, all_blocks_.front().ptr_, all_blocks_.back().ptr_);
  VLOG(1) << "Memory Compact Manual Finish Compact size: " << compact_free_size;

  if (compact_free_size > 0) {
    auto free_block = std::prev(all_blocks_.end());
    assert(free_block->is_free_);
    // remove all old free blocks and put new free block into free_blocks_.
    free_blocks_.clear();
    free_blocks_.emplace(std::make_pair(free_block->size_, free_block->ptr_),
                         free_block);
  }
  return compact_free_size;
}

bool VirtualMemoryAutoGrowthBestFitAllocator::TryAllocateBatch(
    const std::vector<size_t> &sizes) {
  auto SimulateAlloc =
      [&](size_t size,
          std::map<std::pair<size_t, void *>, size_t> &shadow_blocks) {
        auto iter = shadow_blocks.lower_bound(std::make_pair(size, nullptr));
        if (iter != shadow_blocks.end()) {
          size_t block_size = iter->first.first;
          void *block_ptr = iter->first.second;
          shadow_blocks.erase(iter);
          if (NeedSplit(block_size, alignment_, size)) {
            size_t remaining_size = block_size - size;
            void *remaining_ptr = reinterpret_cast<uint8_t *>(block_ptr) + size;
            shadow_blocks.emplace(std::make_pair(remaining_size, remaining_ptr),
                                  remaining_size);
          }
          return true;
        }
        return false;
      };

  std::lock_guard<SpinLock> guard(spinlock_);

  // copy large N free_blocks_ to shadow_blocks_.
  std::map<std::pair<size_t, void *>, size_t> shadow_blocks;
  auto it = free_blocks_.rbegin();
  for (int i = 0; i < sizes.size() && it != free_blocks_.rend(); ++i, ++it) {
    shadow_blocks.emplace(it->first, it->first.first);
  }
  for (size_t size : sizes) {
    size_t aligned_size = AlignedSize(size, alignment_);
    if (!SimulateAlloc(aligned_size, shadow_blocks)) return false;
  }
  return true;
}

std::pair<size_t, size_t>
VirtualMemoryAutoGrowthBestFitAllocator::SumLargestFreeBlockSizes(
    int32_t n) const {
  if (n <= 0 || free_blocks_.empty()) return std::make_pair(0, 0);

  size_t large_size = free_blocks_.rbegin()->first.first;
  size_t total_size = 0;
  int32_t count = 0;

  for (auto it = free_blocks_.rbegin(); it != free_blocks_.rend() && count < n;
       ++it, ++count) {
    total_size += it->first.first;
  }

  return std::make_pair(large_size, total_size);
}

void VirtualMemoryAutoGrowthBestFitAllocator::DumpInfo(
    std::string phase) const {
  size_t total = 0, free = 0, used = 0;
  std::cout << phase << std::endl;
  std::cout << "All_blocks_:" << std::endl;
  for (auto block = all_blocks_.begin(); block != all_blocks_.end(); ++block) {
    std::ostringstream oss_used;
    std::ostringstream oss_free;

    if (block->is_free_) {
      free += block->size_;
      oss_free << "(" << block->size_ << "," << block->ptr_ << ")";
    } else {
      used += block->size_;
      oss_used << "(" << block->size_ << "," << block->ptr_ << ","
               << block->allocation_->ptr() << ")";
    }

    std::cout << "is_free? " << block->is_free_ << "[" << oss_used.str()
              << "]\t[" << oss_free.str() << "]" << std::endl;
  }
  std::cout << total << "\t" << used << "\t" << free << std::endl;
  std::cout << "Free_blocks_:" << std::endl;
  for (const auto &[key, list_iter] : free_blocks_) {
    auto [size, ptr] = key;
    std::cout << "Size: " << size << ", Ptr: " << ptr << "\t" << list_iter->ptr_
              << std::endl;
  }
}

void VirtualMemoryAutoGrowthBestFitAllocator::PreAlloc() {
  auto pre_alloc_size = FLAGS_vmm_pre_alloc_in_mb << 20;
  VLOG(4)
      << "Begin PreAllocate in VirtualMemoryAutoGrowthBestFitAllocator size "
      << pre_alloc_size;
  PreAllocate(pre_alloc_size);
  VLOG(4)
      << "Finish PreAllocate in VirtualMemoryAutoGrowthBestFitAllocator size "
      << pre_alloc_size;
}

void VirtualMemoryAutoGrowthBestFitAllocator::PreAllocate(size_t size) {
  if (size <= 0) return;
  ExtendOrCompact(size);
}

bool VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator::IsSmallRequest(
    size_t size) {
  const size_t routed_size = AlignedSize(size, alignment_);
  const size_t small_pool_size = FLAGS_vmm_small_pool_size_in_mb << 20;
  return routed_size < small_pool_size;
}

void VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator::PreAlloc() {
  auto small_allocator =
      std::dynamic_pointer_cast<VirtualMemoryAutoGrowthBestFitAllocator>(
          GetSmallAllocator());
  auto large_allocator =
      std::dynamic_pointer_cast<VirtualMemoryAutoGrowthBestFitAllocator>(
          GetLargeAllocator());

  auto vmm_small_pool_pre_alloc = FLAGS_vmm_small_pool_pre_alloc_in_mb << 20;
  auto vmm_large_pool_pre_alloc = FLAGS_vmm_large_pool_pre_alloc_in_mb << 20;

  if (vmm_small_pool_pre_alloc > 0 && small_allocator) {
    VLOG(4) << "Begin Small Pool PreAllocate in "
               "VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator size "
            << vmm_small_pool_pre_alloc;
    small_allocator->PreAllocate(vmm_small_pool_pre_alloc);
    VLOG(4) << "Finish Small Pool PreAllocate in "
               "VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator size "
            << vmm_small_pool_pre_alloc;
  }
  if (vmm_large_pool_pre_alloc > 0 && large_allocator) {
    VLOG(4) << "Begin Large Pool PreAllocate in "
               "VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator size "
            << vmm_large_pool_pre_alloc;
    large_allocator->PreAllocate(vmm_large_pool_pre_alloc);
    VLOG(4) << "Finish Large Pool PreAllocate in "
               "VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator size "
            << vmm_large_pool_pre_alloc;
  }
}

size_t VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator::CompactImpl(
    const Place &place) {
  auto large_allocator =
      std::dynamic_pointer_cast<VirtualMemoryAutoGrowthBestFitAllocator>(
          GetLargeAllocator());
  VLOG(1) << "Do Memory Compact Large Pool Manual";
  size_t compact_free_size = large_allocator->Compact(place);
  VLOG(1) << "Memory Compact Large Pool Manual Finish Compact size: "
          << compact_free_size;
  compact_size_.emplace_back(compact_free_size);
  return compact_free_size;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
