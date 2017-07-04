/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/memory/detail/buddy_allocator.h"
#include "glog/logging.h"

namespace paddle {
namespace memory {
namespace detail {

BuddyAllocator::BuddyAllocator(SystemAllocator* system_allocator,
                               size_t min_chunk_size, size_t max_chunk_size) {
  PADDLE_ASSERT(min_chunk_size > 0);
  PADDLE_ASSERT(max_chunk_size > 0);
  PADDLE_ASSERT(system_allocator != nullptr);

  system_allocator_ = std::move(system_allocator);
  min_chunk_size_ = min_chunk_size;
  max_chunk_size_ = max_chunk_size;
}

inline size_t align(size_t size, size_t alignment) {
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

void* BuddyAllocator::Alloc(size_t unaligned_size) {
  // adjust allocation alignment
  size_t size = align(unaligned_size + sizeof(Metadata), min_chunk_size_);

  // acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  DLOG(INFO) << "Allocate " << unaligned_size << " bytes from chunk size "
             << size;

  // if the allocation is huge, send directly to the system allocator
  if (size > max_chunk_size_) {
    DLOG(INFO) << "Allocate from system allocator.";
    return SystemAlloc(size);
  }

  // query and allocate from the existing chunk
  auto it = FindExistChunk(size);

  // refill the pool if failure
  if (it == pool_.end()) {
    it = RefillPool();
    // if still failure, fail fatally
    if (it == pool_.end()) {
      return nullptr;
    }
  } else {
    DLOG(INFO) << " Allocation from existing memory block " << std::get<2>(*it)
               << " at address "
               << reinterpret_cast<MemoryBlock*>(std::get<2>(*it))->data();
  }

  total_used_ += size;
  total_free_ -= size;

  // split the allocation and return data for use
  return reinterpret_cast<MemoryBlock*>(SplitToAlloc(it, size))->data();
}

void BuddyAllocator::Free(void* p) {
  // Point back to metadata
  auto block = static_cast<MemoryBlock*>(p)->metadata();

  // Acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  DLOG(INFO) << "Free from address " << block;

  if (block->type(cache_) == MemoryBlock::HUGE_CHUNK) {
    DLOG(INFO) << "Free directly from system allocator";
    system_allocator_->Free(block, block->total_size(cache_),
                            block->index(cache_));

    // Invalidate GPU allocation from cache
    if (system_allocator_->UseGpu()) {
      cache_.erase(block);
    }
    return;
  }

  block->mark_as_free(cache_);

  total_used_ -= block->total_size(cache_);
  total_free_ += block->total_size(cache_);

  // Trying to merge the right buddy
  if (block->has_right_buddy(cache_)) {
    DLOG(INFO) << "Merging this block " << block << " with its right buddy "
               << block->right_buddy(cache_);
  }

  // Trying to merge the left buddy
  if (block->has_left_buddy(cache_)) {
    DLOG(INFO) << "Merging this block " << block << " with its left buddy "
               << block->left_buddy(cache_);
  }

  // Dumping this block into pool
  DLOG(INFO) << "Inserting free block (" << block << ", "
             << block->total_size(cache_) << ")";
  pool_.insert({block->index(cache_), block->total_size(cache_), block});

  // TODO(gangliao): Clean up if existing too much free memory
}

void* BuddyAllocator::SystemAlloc(size_t size) {
  size_t index = 0;
  void* p = system_allocator_->Alloc(index, size);

  DLOG(INFO) << "Allocated " << p << " from system allocator.";

  if (p == nullptr) return nullptr;

  static_cast<MemoryBlock*>(p)->init(cache_, MemoryBlock::HUGE_CHUNK, index,
                                     size, nullptr, nullptr);

  return static_cast<MemoryBlock*>(p)->data();
}

BuddyAllocator::PoolSet::iterator BuddyAllocator::RefillPool() {
#ifndef PADDLE_ONLY_CPU
  if (system_allocator_->UseGpu()) {
    if ((total_used_ + total_free_) == 0) {
      // Compute the maximum allocation size for the first allocation.
      max_chunk_size_ = platform::GpuMaxChunkSize();
    }
  }
#endif  // PADDLE_ONLY_CPU

  // Allocate a new maximum sized block
  size_t index = 0;
  void* p = system_allocator_->Alloc(index, max_chunk_size_);

  if (p == nullptr) return pool_.end();

  DLOG(INFO) << " Creating and inserting new block " << p
             << " from system allocator";

  static_cast<MemoryBlock*>(p)->init(cache_, MemoryBlock::FREE_CHUNK, index,
                                     max_chunk_size_, nullptr, nullptr);

  total_free_ += max_chunk_size_;

  // dump the block into pool
  return pool_.insert({index, max_chunk_size_, p}).first;
}

BuddyAllocator::PoolSet::iterator BuddyAllocator::FindExistChunk(size_t size) {
  size_t index = 0;

  while (1) {
    auto it = pool_.lower_bound({index, size, nullptr});
    if (it == pool_.end()) return it;

    if (std::get<0>(*it) > index) {
      if (std::get<1>(*it) >= size) {
        return it;
      }

      index = std::get<0>(*it);
      continue;
    }
    return it;
  }
}

void* BuddyAllocator::SplitToAlloc(BuddyAllocator::PoolSet::iterator it,
                                   size_t size) {
  auto block = static_cast<MemoryBlock*>(std::get<2>(*it));
  pool_.erase(it);

  DLOG(INFO) << " Split block (" << block << ", " << block->total_size(cache_)
             << ") into";
  block->split(cache_, size);

  DLOG(INFO) << " Left block (" << block << ", " << block->total_size(cache_)
             << ")";
  block->set_type(cache_, MemoryBlock::ARENA_CHUNK);

  // the rest of memory if exist
  if (block->has_right_buddy(cache_)) {
    if (block->right_buddy(cache_)->type(cache_) == MemoryBlock::FREE_CHUNK) {
      DLOG(INFO) << " Insert right block (" << block->right_buddy(cache_)
                 << ", " << block->right_buddy(cache_)->total_size(cache_)
                 << ")";

      pool_.insert({block->right_buddy(cache_)->index(cache_),
                    block->right_buddy(cache_)->total_size(cache_),
                    block->right_buddy(cache_)});
    }
  }

  return block;
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
