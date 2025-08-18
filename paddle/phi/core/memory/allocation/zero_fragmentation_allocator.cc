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

#include "paddle/phi/core/memory/allocation/zero_fragmentation_allocator.h"

#include <algorithm>
#include <mutex>  // NOLINT
#include <utility>

#include "paddle/common/flags.h"
#include "paddle/phi/backends/device_manager.h"

PHI_DEFINE_EXPORTED_READONLY_bool(
    dump_zero_fragmentation_info,
    false,
    "dump blocks infos of zero fragmentation allocator");

namespace paddle {
namespace memory {
namespace allocation {

phi::Allocation *ZeroFragmentationAllocator::AllocateImpl(
    size_t unaligned_size) {
  size_t size = AlignedSize(unaligned_size + extra_padding_size_, alignment_);
  phi::Allocation *allocation = nullptr;

  if (ZeroFragmentationAllocatorManager::Instance().IsEnabled()) {
    if (ZeroFragmentationAllocatorManager::Instance().IsPeralloc()) {
      FreeZeroFragmentationBlocks();
      allocation = AutoGrowthBestFitAllocator::AllocateImpl(size);
      zero_fragmentation_block_ =
          static_cast<BlockAllocation *>(allocation)->block_it_;
    } else {
      std::lock_guard<SpinLock> guard(spinlock_);
      if (zero_fragmentation_block_ != nulliter &&
          zero_fragmentation_block_->size_ >= size) {
        auto block_it = zero_fragmentation_block_;
        zero_fragmentation_block_ = nulliter;
        auto *chunk = block_it->chunk_;
        size_t remaining_size = block_it->size_ - size;
        VLOG(10) << "Allocate form zero fragmentation pool";
        if (remaining_size == 0) {
          block_it->is_free_ = false;
        } else {
          // Allocate memory in the opposite direction compared to the default
          // pool to reduce backward memory fragmentation.
          auto remaining_free_block = chunk->blocks_.insert(
              std::next(block_it),
              Block(reinterpret_cast<uint8_t *>(block_it->ptr_) + size,
                    remaining_size,
                    false,
                    chunk));
          zero_fragmentation_block_ = remaining_free_block;
          block_it->size_ = size;
          block_it->is_free_ = false;
        }
        allocation = new BlockAllocation(block_it);
      }
    }
  }

  if (allocation == nullptr) {
    VLOG(10) << "Allocate form default pool";
    allocation = AutoGrowthBestFitAllocator::AllocateImpl(size);
  }
  return allocation;
}

void ZeroFragmentationAllocator::FreeImpl(phi::Allocation *allocation) {
  if (ZeroFragmentationAllocatorManager::Instance().IsPeralloc()) {
    HoldZeroFragmentationBlocks(allocation);
  } else {
    AutoGrowthBestFitAllocator::FreeImpl(allocation);
  }
}

uint64_t ZeroFragmentationAllocator::ReleaseImpl(const phi::Place &place) {
  FreeZeroFragmentationBlocks();
  return AutoGrowthBestFitAllocator::ReleaseImpl(place);
}

void ZeroFragmentationAllocator::HoldZeroFragmentationBlocks(
    phi::Allocation *allocation) {
  // Not return to default pool, just reuse the memory.
  std::lock_guard<SpinLock> guard(spinlock_);
  auto block_it = static_cast<BlockAllocation *>(allocation)->block_it_;

  if (zero_fragmentation_block_ != block_it) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ZeroFragmentationAllocator::HoldZeroFragmentationBlocks, "
        "zero_fragmentation_block_ != block_it"));
  }

  block_it->is_free_ = false;

  delete allocation;
}

void ZeroFragmentationAllocator::FreeZeroFragmentationBlocks() {
  std::lock_guard<SpinLock> guard(spinlock_);

  if (zero_fragmentation_block_ == nulliter) {
    return;
  }

  auto block_it = zero_fragmentation_block_;
  block_it->is_free_ = true;

  auto &blocks = block_it->chunk_->blocks_;

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

  if (block_it != (--blocks.end()) && next_it->is_free_) {
    free_blocks_.erase(std::make_pair(next_it->size_, next_it->ptr_));
    block_it->size_ += next_it->size_;
    blocks.erase(next_it);
  }

  free_blocks_.emplace(std::make_pair(block_it->size_, block_it->ptr_),
                       block_it);
  zero_fragmentation_block_ = nulliter;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
