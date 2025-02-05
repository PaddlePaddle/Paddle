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

#include <mutex>

#include "paddle/phi/core/memory/allocation/aligned_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

bool NeedSplit(size_t block_size, size_t alignment, size_t alloc_size) {
  return block_size > (alloc_size * 2) || (block_size - alloc_size) > alignment;
}

VirtualMemoryAutoGrowthBestFitAllocator::
    VirtualMemoryAutoGrowthBestFitAllocator(
        const std::shared_ptr<Allocator> &underlying_allocator,
        size_t alignment,
        const phi::GPUPlace &place)
    : underlying_allocator_(
          std::make_shared<AlignedAllocator>(underlying_allocator, alignment)),
      alignment_(alignment),
      place_(place) {}

phi::Allocation *VirtualMemoryAutoGrowthBestFitAllocator::AllocateImpl(
    size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size = AlignedSize(size, alignment_);
  auto result = AllocFromFreeBlocks(size);

  if (!result) {
    ExtendAndMerge(size);
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

void VirtualMemoryAutoGrowthBestFitAllocator::TryMergeBlock2Blocks(
    std::list<Block>::iterator block) {
  if (block->ptr_ == all_blocks_.front().ptr_ &&
      block->ptr_ == all_blocks_.back().ptr_) {
    block->is_free_ = true;
    free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
  } else if (block->ptr_ == all_blocks_.front().ptr_) {
    auto next = std::next(block);
    if (next->is_free_ &&
        reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ == next->ptr_) {
      // merge with next
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

void VirtualMemoryAutoGrowthBestFitAllocator::ExtendAndMerge(size_t size) {
  void *ptr = nullptr;

  auto allocateptr = underlying_allocator_->Allocate(size);
  ptr = allocateptr->ptr();
  size = allocateptr->size();
  allocations_.push_back(std::move(allocateptr));  // hold allocation

  if (all_blocks_.empty()) {
    all_blocks_.emplace_back(ptr, size, true);
    free_blocks_.emplace(std::make_pair(size, ptr), all_blocks_.begin());
    return;
  }
  for (auto block_it = all_blocks_.begin(); block_it != all_blocks_.end();
       ++block_it) {
    if (block_it->ptr_ > ptr) {
      if (block_it == all_blocks_.begin()) {
        // insert to front
        if (block_it->is_free_ &&
            reinterpret_cast<uint8_t *>(ptr) + size == block_it->ptr_) {
          // merge with next
          free_blocks_.erase(std::make_pair(block_it->size_, block_it->ptr_));
          block_it->ptr_ = ptr;
          block_it->size_ += size;
          free_blocks_.emplace(std::make_pair(block_it->size_, block_it->ptr_),
                               block_it);
        } else {
          // do not merge
          all_blocks_.emplace_back(ptr, size, true);
          free_blocks_.emplace(std::make_pair(size, ptr), all_blocks_.begin());
        }
      } else {
        // insert to middle
        auto next = block_it;
        auto pre = std::prev(block_it);
        if (pre->is_free_ &&
            reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ == ptr &&
            !(next->is_free_ &&
              reinterpret_cast<uint8_t *>(ptr) + size == next->ptr_)) {
          // merge with pre
          free_blocks_.erase(std::make_pair(pre->size_, pre->ptr_));
          pre->size_ += size;
          free_blocks_.emplace(std::make_pair(pre->size_, pre->ptr_), pre);
        } else if (next->is_free_ &&
                   reinterpret_cast<uint8_t *>(ptr) + size == next->ptr_ &&
                   !(pre->is_free_ &&
                     reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ ==
                         ptr)) {
          // merge with next
          free_blocks_.erase(std::make_pair(next->size_, next->ptr_));
          next->ptr_ = ptr;
          next->size_ += size;
          free_blocks_.emplace(std::make_pair(next->size_, next->ptr_), next);
        } else if (pre->is_free_ &&
                   reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ == ptr &&
                   next->is_free_ &&
                   reinterpret_cast<uint8_t *>(ptr) + size == next->ptr_) {
          // merge with pre and next
          free_blocks_.erase(std::make_pair(pre->size_, pre->ptr_));
          free_blocks_.erase(std::make_pair(next->size_, next->ptr_));
          pre->size_ += (size + next->size_);
          free_blocks_.emplace(std::make_pair(pre->size_, pre->ptr_), pre);
          all_blocks_.erase(next);
        } else {
          // do not merge
          auto iter = all_blocks_.insert(next, Block(ptr, size, true));
          free_blocks_.emplace(std::make_pair(size, ptr), iter);
        }
      }
      return;
    }
  }

  // insert to back
  auto block_it = all_blocks_.end();
  block_it--;
  if (block_it->is_free_ &&
      reinterpret_cast<uint8_t *>(block_it->ptr_) + block_it->size_ == ptr) {
    // merge with pre
    free_blocks_.erase(std::make_pair(block_it->size_, block_it->ptr_));
    block_it->size_ += size;
    free_blocks_.emplace(std::make_pair(block_it->size_, block_it->ptr_),
                         block_it);
  } else {
    // do not merge
    all_blocks_.emplace_back(ptr, size, true);
    auto block_it = all_blocks_.end();
    block_it--;
    free_blocks_.emplace(std::make_pair(size, ptr), block_it);
  }
}

phi::Allocation *VirtualMemoryAutoGrowthBestFitAllocator::AllocFromFreeBlocks(
    size_t size) {
  auto iter = free_blocks_.lower_bound(std::make_pair(size, nullptr));
  if (iter != free_blocks_.end()) {
    std::list<Block>::iterator block_it = iter->second;
    free_blocks_.erase(iter);
    if (NeedSplit(block_it->size_, alignment_, size)) {
      size_t remaining_size = block_it->size_ - size;
      auto remaining_free_block = all_blocks_.insert(
          block_it, Block(block_it->ptr_, remaining_size, true));
      free_blocks_.emplace(std::make_pair(remaining_size, block_it->ptr_),
                           remaining_free_block);
      block_it->ptr_ =
          reinterpret_cast<uint8_t *>(block_it->ptr_) + remaining_size;
      block_it->size_ = size;
    }

    block_it->is_free_ = false;
    return new BlockAllocation(block_it, place_);
  }

  return nullptr;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
