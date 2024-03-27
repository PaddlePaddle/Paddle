// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator_v2.h"

#include <algorithm>
#include <mutex>  // NOLINT

#include "paddle/fluid/memory/allocation/aligned_allocator.h"
#include "paddle/fluid/memory/stats.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/flags.h"

DECLARE_bool(free_idle_chunk);
DECLARE_bool(free_when_no_cache_hit);

PADDLE_DEFINE_EXPORTED_bool(
    autogrowth_bestfit_v2_free_idel_when_switch_to_normal,
    false,
    "Whether to free idel when switch to normal auto growth.");

PADDLE_DEFINE_EXPORTED_int64(autogrowth_bestfit_v2_warmup_steps,
                             1,
                             "Warmup step count.");

PADDLE_DEFINE_EXPORTED_int64(autogrowth_bestfit_v2_dbg_level,
                             0,
                             "Print dbg info level.");

PADDLE_DEFINE_EXPORTED_bool(autogrowth_bestfit_v2_stop_warmup_when_mem_full,
                            false,
                            "Whether switch to regular when memory full.");

namespace paddle {
namespace memory {
namespace allocation {

AutoGrowthBestFitAllocatorV2::AutoGrowthBestFitAllocatorV2(
    const std::shared_ptr<Allocator> &underlying_allocator,
    size_t alignment,
    platform::CUDAPlace place,
    size_t chunk_size,
    bool allow_free_idle_chunk)
    : AutoGrowthBestFitAllocator(
          underlying_allocator, alignment, chunk_size, true),
      place_(place) {}

phi::Allocation *AutoGrowthBestFitAllocatorV2::AllocateImpl(
    size_t unaligned_size) {
  platform::RecordEvent record("AutoGrowthBestFitAllocatorV2::Allocate",
                               platform::TracerEventType::UserDefined,
                               9 /*level*/);

  size_t size = AlignedSize(unaligned_size, alignment_);

  VLOG(10) << "Allocate " << unaligned_size << " bytes, aligned to " << size;

  std::lock_guard<SpinLock> guard(spinlock_);

  BlockIt block_it;
  if (!warmup_done_ &&
      AutoGrowthBestFitAllocatorV2State::GetInstance().WarmupCount() <
          FLAGS_autogrowth_bestfit_v2_warmup_steps) {
    auto iter = free_blocks_.lower_bound(std::make_pair(size, nullptr));
    if (iter != free_blocks_.end() && iter->second->size_ >= unaligned_size &&
        iter->second->size_ <= size) {
      block_it = iter->second;
      free_blocks_.erase(iter);
      block_it->is_free_ = false;
      VLOG(10) << "Allocate " << size << " bytes from chunk size "
               << block_it->size_ << " by strict_matching_state.";
    } else {
      try {
        chunks_.emplace_back(static_unique_ptr_cast<Allocation>(
            underlying_allocator_->Allocate(size)));
      } catch (BadAlloc &ex) {
        if (FLAGS_autogrowth_bestfit_v2_stop_warmup_when_mem_full) {
          if (FLAGS_autogrowth_bestfit_v2_dbg_level > 0) {
            std::cout << "warmup mem full" << std::endl;
          }
          warmup_done_ = true;
          throw ex;
        } else {
          FreeIdleChunks();
          chunks_.emplace_back(static_unique_ptr_cast<Allocation>(
              underlying_allocator_->Allocate(size)));
        }
      }

      auto *chunk = &(*chunks_.rbegin());
      size = chunk->allocation_->size();
      uint8_t *p = reinterpret_cast<uint8_t *>(chunk->allocation_->ptr());
      auto &blocks = chunk->blocks_;
      blocks.emplace_back(p, size, false, chunk);
      block_it = --(blocks.end());
      VLOG(2) << "Not found and reallocate " << size << "("
              << static_cast<void *>(p) << ") by strict_matching_state.";
    }
  } else {
    if (is_first_switch_to_regular_) {
      if (FLAGS_autogrowth_bestfit_v2_free_idel_when_switch_to_normal) {
        FreeIdleChunks();
        DEVICE_MEMORY_STAT_RESET_PEAK(Reserved, place_.device);
      }
      if (FLAGS_autogrowth_bestfit_v2_dbg_level > 0) {
        std::cout
            << "switch to regular, warmup_done = " << warmup_done_
            << ", warmup_steps = "
            << AutoGrowthBestFitAllocatorV2State::GetInstance().WarmupCount()
            << std::endl;
        PrintChunks();
      }
      is_first_switch_to_regular_ = false;
    }

    auto iter = free_blocks_.lower_bound(std::make_pair(size, nullptr));

    if (iter != free_blocks_.end()) {
      block_it = iter->second;
      free_blocks_.erase(iter);
      auto *chunk = block_it->chunk_;
      size_t remaining_size = block_it->size_ - size;
      VLOG(10) << "Allocate " << size << " bytes from chunk size "
               << block_it->size_ << ", remaining " << remaining_size;
      if (remaining_size == 0) {
        block_it->is_free_ = false;
      } else {
        auto remaining_free_block = chunk->blocks_.insert(
            block_it, Block(block_it->ptr_, remaining_size, true, chunk));
        free_blocks_.emplace(std::make_pair(remaining_size, block_it->ptr_),
                             remaining_free_block);
        block_it->ptr_ =
            reinterpret_cast<uint8_t *>(block_it->ptr_) + remaining_size;
        block_it->size_ = size;
        block_it->is_free_ = false;
      }
    } else {
      size_t realloc_size = std::max(size, chunk_size_);
      if (FLAGS_autogrowth_bestfit_v2_dbg_level > 0) {
        std::cout << "auto growth try alloc " << realloc_size << std::endl;
        PrintChunks();
      }

      try {
        chunks_.emplace_back(static_unique_ptr_cast<Allocation>(
            underlying_allocator_->Allocate(realloc_size)));
      } catch (BadAlloc &ex) {
        if (FLAGS_autogrowth_bestfit_v2_dbg_level > 0) {
          std::cout << "auto growth try alloc OOM, try free idel" << std::endl;
        }
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
        blocks.emplace_back(p, remaining_size, true, chunk);
        free_blocks_.emplace(std::make_pair(remaining_size, p),
                             --(blocks.end()));
      }
      blocks.emplace_back(p + remaining_size, size, false, chunk);
      block_it = --(blocks.end());
      VLOG(2) << "Not found and reallocate " << realloc_size << "("
              << static_cast<void *>(p) << "), and remaining "
              << remaining_size;
    }
  }
  ++total_alloc_times_;
  total_alloc_size_ += size;
  VLOG(10) << "Alloc " << block_it->size_ << " bytes, ptr = " << block_it->ptr_;
  if (FLAGS_autogrowth_bestfit_v2_dbg_level > 1) {
    std::cout << "Alloc " << block_it->size_
              << " bytes, ptr = " << block_it->ptr_ << std::endl;
  }

  return new BlockAllocation(block_it);
}

void AutoGrowthBestFitAllocatorV2::FreeImpl(phi::Allocation *allocation) {
  platform::RecordEvent record("AutoGrowthBestFitAllocatorV2::Free",
                               platform::TracerEventType::UserDefined,
                               9 /*level*/);
  VLOG(10) << "Free " << allocation->size()
           << " bytes, ptr = " << allocation->ptr();
  if (FLAGS_autogrowth_bestfit_v2_dbg_level > 1) {
    std::cout << "Free " << allocation->size()
              << " bytes, ptr = " << allocation->ptr() << std::endl;
  }

  std::lock_guard<SpinLock> guard(spinlock_);
  auto block_it = static_cast<BlockAllocation *>(allocation)->block_it_;
  auto &blocks = block_it->chunk_->blocks_;

  total_free_times_ += 1;
  total_free_size_ += block_it->size_;

  block_it->is_free_ = true;

  if (block_it != blocks.begin()) {
    auto prev_it = block_it;
    --prev_it;

    if (prev_it->is_free_) {
      free_blocks_.erase(std::make_pair(prev_it->size_, prev_it->ptr_));
      prev_it->size_ += block_it->size_;
      blocks.erase(block_it);
      block_it = prev_it;
    }
  }

  auto next_it = block_it;
  ++next_it;

  if (next_it != blocks.end() && next_it->is_free_) {
    free_blocks_.erase(std::make_pair(next_it->size_, next_it->ptr_));
    block_it->size_ += next_it->size_;
    blocks.erase(next_it);
  }

  free_blocks_.emplace(std::make_pair(block_it->size_, block_it->ptr_),
                       block_it);

  delete allocation;
}

void AutoGrowthBestFitAllocatorV2::PrintChunks() {
  size_t totol_free = 0;
  for (auto &c : chunks_) {
    std::stringstream ss_f;
    std::stringstream ss_nf;
    size_t tmp = 0;
    size_t tmp2 = 0;
    for (auto &l : c.blocks_) {
      if (l.is_free_) {
        ss_f << "(" << l.ptr_ << "," << l.size_ << "),";
        tmp += l.size_;
      } else {
        ss_nf << "(" << l.ptr_ << "," << l.size_ << "),";
        tmp2 += l.size_;
      }
    }
    totol_free += tmp;
    std::cout << "Chunk FreeSize: " << tmp << ", "
              << "AllocedSize: " << tmp2 << ", Free: " << ss_f.str()
              << "Alloced: " << ss_nf.str() << "Chunk end";
  }
  std::cout << std::endl;
  std::cout << "totol_free = " << totol_free << std::endl;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
#endif
