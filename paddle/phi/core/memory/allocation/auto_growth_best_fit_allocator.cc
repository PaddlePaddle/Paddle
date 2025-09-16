// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/auto_growth_best_fit_allocator.h"

#include <algorithm>
#include <mutex>  // NOLINT
#include <utility>

#include "paddle/common/flags.h"
#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/memory/allocation/aligned_allocator.h"

PHI_DEFINE_EXPORTED_READONLY_bool(
    free_idle_chunk,
    false,
    "Whether to free idle chunk when each allocation is freed. "
    "If false, all freed allocation would be cached to speed up next "
    "allocation request. If true, no allocation would be cached. This "
    "flag only works when FLAGS_allocator_strategy=auto_growth.");

PHI_DEFINE_EXPORTED_READONLY_bool(
    free_when_no_cache_hit,
    false,
    "Whether to free idle chunks when no cache hit. If true, idle "
    "chunk would be freed when no cache hit; if false, idle "
    "chunk would be freed when out of memory occurs. This flag "
    "only works when FLAGS_allocator_strategy=auto_growth.");

PHI_DEFINE_EXPORTED_READONLY_bool(print_allocator_trace_info,
                                  false,
                                  "print trace memory info");

PHI_DEFINE_EXPORTED_READONLY_bool(dump_chunk_info, false, "dump chunk info");
PHI_DEFINE_EXPORTED_uint64(
    alignment_size,
    256,
    "All sizes are rounded up to a multiple of this value. Default: 256.");
PHI_DEFINE_EXPORTED_uint64(
    small_pool_size_in_mb,
    0,
    "Threshold (MiB) separating the small and large pools. "
    "0 disables the small pool and enables single-pool mode "
    "(all requests go to the large pool). When > 0, requests "
    "<= threshold use the small pool; larger requests use the "
    "large pool. Default: 0.");
PHI_DEFINE_EXPORTED_uint64(small_pool_auto_growth_chunk_size_in_mb,
                           0,
                           "The minimal chunk size for the small pool in MiB. "
                           "If small_pool_size_in_mb > 0, this overrides "
                           "the constructor-provided global growth size "
                           "(FLAGS_auto_growth_chunk_size_in_mb).");
PHI_DEFINE_EXPORTED_uint64(large_pool_auto_growth_chunk_size_in_mb,
                           0,
                           "The minimal chunk size for the large pool in MiB. "
                           "If small_pool_size_in_mb > 0, this overrides "
                           "the constructor-provided global growth size "
                           "(FLAGS_auto_growth_chunk_size_in_mb).");
PHI_DEFINE_EXPORTED_uint64(
    large_pool_pre_alloc_in_mb,
    0,
    "Pre-reserve this many MiB in the large pool. 0 disables pre-allocation.");
PHI_DEFINE_EXPORTED_uint64(
    small_pool_pre_alloc_in_mb,
    0,
    "Pre-reserve this many MiB in the small pool. 0 disables pre-allocation.");

namespace paddle::memory::allocation {

AutoGrowthBestFitAllocator::AutoGrowthBestFitAllocator(
    std::shared_ptr<Allocator> underlying_allocator,
    size_t alignment,
    size_t chunk_size,
    bool allow_free_idle_chunk,
    int extra_padding_size)
    : underlying_allocator_(std::move(underlying_allocator)),
      alignment_(alignment),
      chunk_size_(std::max(AlignedSize(chunk_size, alignment), alignment)),
      allow_free_idle_chunk_(allow_free_idle_chunk),
      extra_padding_size_(extra_padding_size) {
  total_alloc_times_ = 0;
  total_alloc_size_ = 0;
  total_free_times_ = 0;
  total_free_size_ = 0;
  VLOG(4) << "chunk_size_:" << chunk_size_;
}

void AutoGrowthBestFitAllocator::DumpInfo() const {
  for (auto chunk_it = chunks_.begin(); chunk_it != chunks_.end(); ++chunk_it) {
    std::cout << "Chunk\t";
    std::ostringstream oss_used;
    std::ostringstream oss_free;
    size_t total = 0, free = 0, used = 0;
    for (auto &b : chunk_it->blocks_) {
      total += b.size_;
      if (b.is_free_) {
        free += b.size_;
        oss_free << "(" << b.size_ << "," << b.ptr_ << ")";
      } else {
        used += b.size_;
        oss_used << "(" << b.size_ << "," << b.ptr_ << ")";
      }
    }
    std::cout << total << "\t" << used << "\t" << free << "\t";
    std::cout << "[" << oss_used.str() << "]\t[" << oss_free.str() << "]"
              << std::endl;
  }
}

bool AutoGrowthBestFitAllocator::is_small_free_block(size_t size) {
  auto small_pool_size = FLAGS_small_pool_size_in_mb << 20;
  if (size <= small_pool_size) {
    return true;
  } else {
    return false;
  }
}

size_t AutoGrowthBestFitAllocator::auto_growth_size(bool is_small,
                                                    size_t chunk_size) {
  // fallback to single pool and use constructor-provided chunk_size.
  if (FLAGS_small_pool_size_in_mb == 0) {
    return chunk_size;
  }

  const uint64_t pool_auto_growth_chunk_size_mb =
      is_small ? FLAGS_small_pool_auto_growth_chunk_size_in_mb
               : FLAGS_large_pool_auto_growth_chunk_size_in_mb;
  const size_t auto_growth_size =
      pool_auto_growth_chunk_size_mb
          ? (static_cast<size_t>(pool_auto_growth_chunk_size_mb) << 20)
          : 0;

  return AlignedSize(auto_growth_size, alignment_);
}

void AutoGrowthBestFitAllocator::PreAlloc() {
  auto small_pool_pre_alloc = FLAGS_small_pool_pre_alloc_in_mb << 20;
  auto large_pool_pre_alloc = FLAGS_large_pool_pre_alloc_in_mb << 20;
  if (small_pool_pre_alloc > 0) {
    VLOG(10) << "PreAlloc small_pool_pre_alloc_in_mb = "
             << FLAGS_small_pool_pre_alloc_in_mb;
    chunks_.emplace_back(static_unique_ptr_cast<Allocation>(
        underlying_allocator_->Allocate(small_pool_pre_alloc)));
    auto *chunk = &(*chunks_.rbegin());
    uint8_t *p = reinterpret_cast<uint8_t *>(chunk->allocation_->ptr());
    auto &blocks = chunk->blocks_;
    blocks.emplace_back(
        p, small_pool_pre_alloc, /*is_free=*/true, /*is_small=*/true, chunk);
    small_free_blocks_.emplace(std::make_pair(small_pool_pre_alloc, p),
                               --(blocks.end()));
  }

  if (large_pool_pre_alloc > 0) {
    VLOG(10) << "PreAlloc large_pool_pre_alloc_in_mb = "
             << FLAGS_large_pool_pre_alloc_in_mb;
    chunks_.emplace_back(static_unique_ptr_cast<Allocation>(
        underlying_allocator_->Allocate(large_pool_pre_alloc)));
    auto *chunk = &(*chunks_.rbegin());
    uint8_t *p = reinterpret_cast<uint8_t *>(chunk->allocation_->ptr());
    auto &blocks = chunk->blocks_;
    blocks.emplace_back(
        p, large_pool_pre_alloc, /*is_free=*/true, /*is_small=*/false, chunk);
    large_free_blocks_.emplace(std::make_pair(large_pool_pre_alloc, p),
                               --(blocks.end()));
  }
}

phi::Allocation *AutoGrowthBestFitAllocator::AllocateImpl(
    size_t unaligned_size) {
  phi::RecordEvent record("AutoGrowthBestFitAllocator::Allocate",
                          phi::TracerEventType::UserDefined,
                          9 /*level*/);

  size_t size = AlignedSize(unaligned_size + extra_padding_size_, alignment_);

  VLOG(10) << "Allocate " << unaligned_size << " bytes, aligned to " << size
           << ", extra size " << extra_padding_size_;

  std::lock_guard<SpinLock> guard(spinlock_);
  bool is_small = is_small_free_block(size);
  auto &free_blocks = is_small ? small_free_blocks_ : large_free_blocks_;
  auto iter = free_blocks.lower_bound(std::make_pair(size, nullptr));
  BlockIt block_it;
  if (iter != free_blocks.end()) {
    block_it = iter->second;
    free_blocks.erase(iter);
    auto *chunk = block_it->chunk_;
    size_t remaining_size = block_it->size_ - size;
    VLOG(10) << "Allocate " << size << " bytes from chunk size "
             << block_it->size_ << ", remaining " << remaining_size;
    if (remaining_size == 0) {
      block_it->is_free_ = false;
      block_it->is_small_ = is_small;
    } else {
      auto remaining_free_block = chunk->blocks_.insert(
          block_it,
          Block(block_it->ptr_, remaining_size, true, is_small, chunk));
      free_blocks.emplace(std::make_pair(remaining_size, block_it->ptr_),
                          remaining_free_block);
      block_it->ptr_ =
          reinterpret_cast<uint8_t *>(block_it->ptr_) + remaining_size;
      block_it->size_ = size;
      block_it->is_free_ = false;
      block_it->is_small_ = is_small;
    }
  } else {
    if (FLAGS_dump_chunk_info) {
      std::cout << "MemDbg memory not enough growth chunk, need size = " << size
                << std::endl;
      DumpInfo();
    }

    if (FLAGS_free_when_no_cache_hit) {
      FreeIdleChunks();
    }
    size_t realloc_size =
        std::max(size, auto_growth_size(is_small, chunk_size_));

    try {
      chunks_.emplace_back(static_unique_ptr_cast<Allocation>(
          underlying_allocator_->Allocate(realloc_size)));
    } catch (BadAlloc &ex) {
      if (FLAGS_dump_chunk_info) {
        std::cout << "MemDbg OOM" << std::endl;
        DumpInfo();
      }
      if (FLAGS_free_when_no_cache_hit) throw ex;
      FreeIdleChunks();
      chunks_.emplace_back(static_unique_ptr_cast<Allocation>(
          underlying_allocator_->Allocate(realloc_size)));
    }

    auto *chunk = &(*chunks_.rbegin());
    realloc_size = chunk->allocation_->size();
    uint8_t *p = reinterpret_cast<uint8_t *>(chunk->allocation_->ptr());
    auto &blocks = chunk->blocks_;

    size_t remaining_size = realloc_size - size;
    if (remaining_size > 0) {
      blocks.emplace_back(p, remaining_size, true, is_small, chunk);
      free_blocks.emplace(std::make_pair(remaining_size, p), --(blocks.end()));
    }
    blocks.emplace_back(p + remaining_size, size, false, is_small, chunk);
    block_it = --(blocks.end());
    VLOG(2) << "Not found and reallocate " << realloc_size << "("
            << static_cast<void *>(p) << "), and remaining " << remaining_size;
    if (FLAGS_dump_chunk_info) {
      std::cout << "MemDbg memory after growth chunk, realloc_size = "
                << realloc_size << std::endl;
      DumpInfo();
    }
  }
  ++total_alloc_times_;
  total_alloc_size_ += size;
  VLOG(10) << "Alloc " << block_it->size_ << " bytes, ptr = " << block_it->ptr_;
  auto block_t = new BlockAllocation(block_it);
  return block_t;
}

void AutoGrowthBestFitAllocator::FreeImpl(phi::Allocation *allocation) {
  phi::RecordEvent record("AutoGrowthBestFitAllocator::Free",
                          phi::TracerEventType::UserDefined,
                          9 /*level*/);
  VLOG(10) << "Free " << allocation->size()
           << " bytes, ptr = " << allocation->ptr();
  std::lock_guard<SpinLock> guard(spinlock_);
  auto block_it = static_cast<BlockAllocation *>(allocation)->block_it_;
  auto &blocks = block_it->chunk_->blocks_;
  bool is_small = block_it->is_small_;
  auto &free_blocks = is_small ? small_free_blocks_ : large_free_blocks_;

  total_free_times_ += 1;
  total_free_size_ += block_it->size_;

  block_it->is_free_ = true;

  if (block_it != blocks.begin()) {
    auto prev_it = block_it;
    --prev_it;

    if (prev_it->is_free_) {
      free_blocks.erase(std::make_pair(prev_it->size_, prev_it->ptr_));
      prev_it->size_ += block_it->size_;
      blocks.erase(block_it);
      block_it = prev_it;
    }
  }

  auto next_it = block_it;
  ++next_it;

  // It's weird that using `next_it == blocks.end()` will cause a judgment fail.
  if (block_it != (--blocks.end()) && next_it->is_free_) {
    free_blocks.erase(std::make_pair(next_it->size_, next_it->ptr_));
    block_it->size_ += next_it->size_;
    blocks.erase(next_it);
  }

  free_blocks.emplace(std::make_pair(block_it->size_, block_it->ptr_),
                      block_it);

  delete allocation;

  if (FLAGS_free_idle_chunk) {
    FreeIdleChunks();
  }
  if (FLAGS_dump_chunk_info) {
    DumpInfo();
  }
}

uint64_t AutoGrowthBestFitAllocator::FreeIdleChunks() {
  if (FLAGS_dump_chunk_info) {
    std::cout << "FreeIdleChunks called" << std::endl;
  }
  if (!allow_free_idle_chunk_) {
    return 0;
  }
  uint64_t bytes = 0;
  for (auto chunk_it = chunks_.begin(); chunk_it != chunks_.end();) {
    auto &blocks = chunk_it->blocks_;
    if (blocks.size() == 1 && blocks.begin()->is_free_) {
      auto &block = *blocks.begin();
      bool is_small = block.is_small_;
      auto &free_blocks = is_small ? small_free_blocks_ : large_free_blocks_;
      VLOG(2) << "Free chunk with size " << block.size_;
      if (FLAGS_dump_chunk_info) {
        std::cout << "FreeIdleChunks chunk is " << block.size_ << ", "
                  << block.ptr_ << std::endl;
      }
      bytes += block.size_;
      free_blocks.erase(std::make_pair(block.size_, block.ptr_));
      chunk_it = chunks_.erase(chunk_it);
    } else {
      ++chunk_it;
    }
  }

  if (FLAGS_print_allocator_trace_info) {
    Trace();
  }
  return bytes;
}

void AutoGrowthBestFitAllocator::Trace() const {
  size_t small_cur_idle_bytes = 0;
  auto small_it = small_free_blocks_.begin();
  for (; small_it != small_free_blocks_.end(); ++small_it) {
    small_cur_idle_bytes += small_it->second->size_;
  }
  size_t large_cur_idle_bytes = 0;
  auto large_it = large_free_blocks_.begin();
  for (; large_it != large_free_blocks_.end(); ++large_it) {
    large_cur_idle_bytes += large_it->second->size_;
  }

  VLOG(1) << "alloc:"
          << total_alloc_size_ / static_cast<double>(1024 * 1024)  // NOLINT
          << "m free:"
          << total_free_size_ / static_cast<double>(1024 * 1024)  // NOLINT
          << "m busy:"
          << (total_alloc_size_ - total_free_size_) /  // NOLINT
                 static_cast<double>(1024 * 1024)
          << "m small idle:"
          << small_cur_idle_bytes / static_cast<double>(1024 * 1024)  // NOLINT
          << "m large idle:"
          << large_cur_idle_bytes / static_cast<double>(1024 * 1024)  // NOLINT
          << "m alloc_times:" << total_alloc_times_
          << " free_times:" << total_free_times_
          << " small free_blocks_num:" << small_free_blocks_.size()
          << " large free_blocks_num:" << large_free_blocks_.size()
          << " curr_chunks_num:" << chunks_.size();
}

}  // namespace paddle::memory::allocation
