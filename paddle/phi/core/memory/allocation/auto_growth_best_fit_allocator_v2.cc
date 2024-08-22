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
#include "paddle/phi/core/memory/allocation/auto_growth_best_fit_allocator_v2.h"

#include <algorithm>
#include <mutex>  // NOLINT

#include "paddle/common/flags.h"
#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/memory/allocation/aligned_allocator.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

PD_DECLARE_bool(free_idle_chunk);
PD_DECLARE_bool(free_when_no_cache_hit);

namespace paddle::memory::allocation {

AutoGrowthBestFitAllocatorV2::AutoGrowthBestFitAllocatorV2(
    const std::shared_ptr<Allocator> &underlying_allocator,
    size_t alignment,
    phi::GPUPlace place,
    size_t chunk_size,
    bool allow_free_idle_chunk,
    int extra_padding_size)
    : AutoGrowthBestFitAllocator(underlying_allocator,
                                 alignment,
                                 chunk_size,
                                 true,
                                 extra_padding_size),
      place_(place) {}

phi::Allocation *AutoGrowthBestFitAllocatorV2::AllocateImpl(
    size_t unaligned_size) {
  phi::RecordEvent record("AutoGrowthBestFitAllocatorV2::Allocate",
                          phi::TracerEventType::UserDefined,
                          9 /*level*/);

  size_t size = AlignedSize(unaligned_size + extra_padding_size_, alignment_);

  VLOG(10) << "Allocate " << unaligned_size << " bytes, aligned to " << size
           << ", extra size " << extra_padding_size_;

  std::lock_guard<SpinLock> guard(spinlock_);

  BlockIt block_it;
  if (AutoGrowthBestFitAllocatorV2State::GetInstance().IsWarmup()) {
    auto iter = free_blocks_.lower_bound(std::make_pair(size, nullptr));
    if (iter != free_blocks_.end() && iter->second->size_ >= unaligned_size &&
        iter->second->size_ <= size) {
      block_it = iter->second;
      free_blocks_.erase(iter);
      block_it->is_free_ = false;
      VLOG(10) << "Allocate " << size << " bytes from chunk size "
               << block_it->size_ << " by strict_matching_state.";
    } else {
      size_t actual_avail, actual_total;
      {
        platform::CUDADeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_HIP
        auto result = hipMemGetInfo(&actual_avail, &actual_total);
#else
        auto result = cudaMemGetInfo(&actual_avail, &actual_total);
#endif
        if (result != gpuSuccess) {
          actual_avail = 0;
        }
      }

      if (actual_avail < size) {
        FreeIdleChunks();
      }

      chunks_.emplace_back(static_unique_ptr_cast<Allocation>(
          underlying_allocator_->Allocate(size)));

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
      FreeIdleChunks();
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
      if (FLAGS_free_when_no_cache_hit) {
        FreeIdleChunks();
      }
      size_t realloc_size = std::max(size, chunk_size_);

      try {
        chunks_.emplace_back(static_unique_ptr_cast<Allocation>(
            underlying_allocator_->Allocate(realloc_size)));
      } catch (BadAlloc &ex) {
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
  return new BlockAllocation(block_it);
}

}  // namespace paddle::memory::allocation
#endif
