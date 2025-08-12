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
      if (zero_fragmentation_block_ != nulliter) {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "ZeroFragmentationAllocator::AllocateImpl, already has a buffer "
            "block"));
      } else {
        allocation = AutoGrowthBestFitAllocator::AllocateImpl(size);
        zero_fragmentation_block_ =
            static_cast<BlockAllocation *>(allocation)->block_it_;
        zero_fragmentation_block_->in_default_pool_ = false;
      }
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
                    true,
                    false,
                    chunk));
          zero_fragmentation_block_ = remaining_free_block;
          block_it->size_ = size;
          block_it->is_free_ = false;
          block_it->in_default_pool_ = true;
        }

        ++total_alloc_times_;
        total_alloc_size_ += size;
        allocation = new BlockAllocation(block_it);
      }
    }
  }

  if (allocation == nullptr) {
    VLOG(10) << "Allocate form default pool";
    try {
      allocation = AutoGrowthBestFitAllocator::AllocateImpl(size);
    } catch (BadAlloc &ex) {
      VLOG(10) << "ZeroFragmentationAllocator MemDbg OOM";
      DeallocateZeroFragmentationBlocks();
      if (FLAGS_dump_zero_fragmentation_info) {
        DumpInfo();
      }
      allocation = AutoGrowthBestFitAllocator::AllocateImpl(size);
    }
  }
  return allocation;
}

void ZeroFragmentationAllocator::FreeImpl(phi::Allocation *allocation) {
  if (ZeroFragmentationAllocatorManager::Instance().IsPeralloc()) {
    FreeZeroFragmentationBlocks(allocation);
  } else if (ZeroFragmentationAllocatorManager::Instance().IsDeallocate()) {
    AutoGrowthBestFitAllocator::FreeImpl(allocation);
    DeallocateZeroFragmentationBlocks();
  } else {
    AutoGrowthBestFitAllocator::FreeImpl(allocation);
  }
}

void ZeroFragmentationAllocator::FreeZeroFragmentationBlocks(
    phi::Allocation *allocation) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto block_it = static_cast<BlockAllocation *>(allocation)->block_it_;

  if (zero_fragmentation_block_ != block_it) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "ZeroFragmentationAllocator::FreeZeroFragmentationBlocks, "
        "zero_fragmentation_block_ != block_it"));
  }

  total_free_times_ += 1;
  total_free_size_ += block_it->size_;

  block_it->is_free_ = true;

  delete allocation;
}

uint64_t ZeroFragmentationAllocator::FreeIdleChunks() {
  DeallocateZeroFragmentationBlocksUnsafe();
  return AutoGrowthBestFitAllocator::FreeIdleChunks();
}

void ZeroFragmentationAllocator::DeallocateZeroFragmentationBlocks() {
  std::lock_guard<SpinLock> guard(spinlock_);
  DeallocateZeroFragmentationBlocksUnsafe();
}
void ZeroFragmentationAllocator::DeallocateZeroFragmentationBlocksUnsafe() {
  if (zero_fragmentation_block_ == nulliter) {
    return;
  }

  auto block_it = zero_fragmentation_block_;
  auto &blocks = block_it->chunk_->blocks_;
  block_it->in_default_pool_ = true;

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

  // It's weird that using `next_it == blocks.end()` will cause a judgment fail.
  if (block_it != (--blocks.end()) && next_it->is_free_) {
    free_blocks_.erase(std::make_pair(next_it->size_, next_it->ptr_));
    block_it->size_ += next_it->size_;
    blocks.erase(next_it);
    block_it->in_default_pool_ = true;
  }

  free_blocks_.emplace(std::make_pair(block_it->size_, block_it->ptr_),
                       block_it);
  zero_fragmentation_block_ = nulliter;
}

void ZeroFragmentationAllocator::DumpInfo() const {
  std::cout << "---Start DumpInfo---" << std::endl;
  for (auto chunk_it = chunks_.begin(); chunk_it != chunks_.end(); ++chunk_it) {
    std::cout << "Chunk\t";
    std::ostringstream oss_used;
    std::ostringstream oss_free;
    std::ostringstream oss_blocks;
    size_t total = 0, free = 0, used = 0;
    for (auto &b : chunk_it->blocks_) {
      total += b.size_;
      if (b.is_free_) {
        free += b.size_;
        oss_free << "(" << b.size_ << ", " << b.ptr_
                 << ", in_default_pool=" << b.in_default_pool_ << ")";
      } else {
        used += b.size_;
        oss_used << "(" << b.size_ << ", " << b.ptr_
                 << ", in_default_pool=" << b.in_default_pool_ << ")";
      }
    }
    std::cout << "total:" << total << "\t"
              << "used:" << used << "\t"
              << "free:" << free << std::endl;
    std::cout << "used:[" << oss_used.str() << "]\nfreed:[" << oss_free.str()
              << "]" << std::endl;
    oss_blocks << "zero_fragmentation_block:";
    if (zero_fragmentation_block_ != nulliter) {
      oss_blocks << "(" << zero_fragmentation_block_->size_ << ", "
                 << zero_fragmentation_block_->ptr_ << ", in_default_pool="
                 << zero_fragmentation_block_->in_default_pool_ << ")\n";
    } else {
      oss_blocks << "(nulliter)\n";
    }
    oss_blocks << "free_blocks:";
    for (auto &pair : free_blocks_) {
      oss_blocks << "(" << pair.second->size_ << ", " << pair.second->ptr_
                 << ", in_default_pool=" << pair.second->in_default_pool_
                 << ")";
    }
    std::cout << oss_blocks.str() << std::endl;
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
