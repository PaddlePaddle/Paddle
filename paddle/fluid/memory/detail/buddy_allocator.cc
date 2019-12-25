/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/detail/buddy_allocator.h"

#include <algorithm>
#include <utility>

#include "glog/logging.h"

#ifdef PADDLE_WITH_CUDA
DECLARE_uint64(reallocate_gpu_memory_in_mb);
#endif

namespace paddle {
namespace memory {
namespace detail {

BuddyAllocator::BuddyAllocator(
    std::unique_ptr<SystemAllocator> system_allocator, size_t min_chunk_size,
    size_t max_chunk_size)
    : min_chunk_size_(min_chunk_size),
      max_chunk_size_(max_chunk_size),
      system_allocator_(std::move(system_allocator)) {
  ptr_to_block_.reserve(200);
}

BuddyAllocator::~BuddyAllocator() {
  VLOG(10) << "BuddyAllocator Disconstructor makes sure that all of these "
              "have actually been freed";
  // free unmerged blocks
  for (auto it = ptr_to_block_.begin(); it != ptr_to_block_.end(); ++it) {
    Free(it->first);
  }

  for (auto& pool : pools_)
    while (!pool.empty()) {
      auto block = *pool.begin();
      VLOG(10) << "Free from block (" << block->get_data() << ", "
               << block->get_size() << ")";

      system_allocator_->Free(block->get_data(), block->get_size(),
                              block->get_index());
      pool.erase(pool.begin());
    }
}

inline size_t align(size_t size, size_t alignment) {
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

void* BuddyAllocator::Alloc(size_t unaligned_size) {
  // adjust allocation alignment
  size_t size = align(unaligned_size, min_chunk_size_);

  // acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  VLOG(10) << "Allocate " << unaligned_size << " bytes from chunk size "
           << size;

  // if the allocation is huge, send directly to the system allocator
  if (size > max_chunk_size_) {
    VLOG(10) << "Allocate from system allocator.";
    return SystemAlloc(size);
  }

  // query and allocate from the existing chunk
  auto ptr = FindExistChunk(size);

  if (!ptr) return nullptr;

  VLOG(10) << "Allocation from existing memory block " << ptr << " at address "
           << ptr->get_data();

  total_used_ += size;
  total_free_ -= size;

  // split the allocation and return data for use
  return SplitToAlloc(ptr, size)->get_data();
}

void BuddyAllocator::Free(void* p) {
  // Point back to metadata
  auto block = ptr_to_block_.at(p);
  ptr_to_block_.erase(p);

  // Acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  VLOG(10) << "Free from address " << block;

  if (block->get_type() == MemoryBlock::HUGE_CHUNK) {
    VLOG(10) << "Free directly from system allocator";
    system_allocator_->Free(block->get_data(), block->get_size(),
                            block->get_index());

    mb_pool_.Push(block);

    return;
  }

  block->MarkAsFree();

  total_used_ -= block->get_size();
  total_free_ += block->get_size();

  // Trying to merge the right buddy
  MemoryBlock* right_buddy = block->get_right_buddy();
  if (right_buddy) {
    VLOG(10) << "Merging this block " << block->get_data()
             << " with its right buddy " << right_buddy->get_data();

    if (right_buddy->get_type() == MemoryBlock::FREE_CHUNK) {
      // Take away right buddy from pool
      pools_[right_buddy->get_index()].erase(right_buddy);

      // merge its right buddy to the block
      block->Merge(right_buddy, &mb_pool_);
    }
  }

  // Trying to merge the left buddy
  MemoryBlock* left_buddy = block->get_left_buddy();
  if (left_buddy) {
    VLOG(10) << "Merging this block " << block->get_data()
             << " with its left buddy " << left_buddy->get_data();

    if (left_buddy->get_type() == MemoryBlock::FREE_CHUNK) {
      // Take away left buddy from pool
      pools_[left_buddy->get_index()].erase(left_buddy);

      // merge the block to its left buddy
      left_buddy->Merge(block, &mb_pool_);
      block = left_buddy;
    }
  }

  // Dumping this block into pool
  VLOG(10) << "Inserting free block (" << block->get_data() << ", "
           << block->get_size() << ")";
  pools_[block->get_index()].insert(block);
}

size_t BuddyAllocator::Used() { return total_used_; }
size_t BuddyAllocator::GetMinChunkSize() { return min_chunk_size_; }
size_t BuddyAllocator::GetMaxChunkSize() { return max_chunk_size_; }

void* BuddyAllocator::SystemAlloc(size_t size) {
  size_t index = 0;
  void* p = system_allocator_->Alloc(&index, size);

  VLOG(10) << "Allocated " << p << " from system allocator.";

  if (p == nullptr) return nullptr;

  // auto block = new MemoryBlock(p, MemoryBlock::HUGE_CHUNK, index, size,
  // nullptr, nullptr);
  auto block = mb_pool_.Get();
  block->Init(p, MemoryBlock::HUGE_CHUNK, index, size, nullptr, nullptr);
  ptr_to_block_.insert({p, block});

  return p;
}

MemoryBlock* BuddyAllocator::RefillPool(size_t request_bytes) {
  size_t allocate_bytes = max_chunk_size_;
  size_t index = 0;

#ifdef PADDLE_WITH_CUDA
  if (system_allocator_->UseGpu()) {
    if ((total_used_ + total_free_) == 0) {
      // Compute the allocation size for gpu for the first allocation.
      allocate_bytes = std::max(platform::GpuInitAllocSize(), request_bytes);
    } else {
      // Compute the re-allocation size, we store the re-allocation size when
      // user set FLAGS_reallocate_gpu_memory_in_mb to fix value.
      if (realloc_size_ == 0 || FLAGS_reallocate_gpu_memory_in_mb == 0ul) {
        realloc_size_ = platform::GpuReallocSize();
      }
      allocate_bytes = std::max(realloc_size_, request_bytes);
    }
  }
#endif

  // Allocate a new block
  void* p = system_allocator_->Alloc(&index, allocate_bytes);

  // return nullptr if system_allocator failed to allocate new memory
  if (p == nullptr) return nullptr;

  VLOG(10) << "Creating and inserting new block " << p
           << " from system allocator";

  // auto block = new MemoryBlock(p, MemoryBlock::FREE_CHUNK, index,
  // allocate_bytes, nullptr, nullptr);
  auto block = mb_pool_.Get();
  block->Init(p, MemoryBlock::FREE_CHUNK, index, allocate_bytes, nullptr,
              nullptr);

  total_free_ += allocate_bytes;

  return block;
}

MemoryBlock* BuddyAllocator::FindExistChunk(size_t size) {
  // search in pools
  MemoryBlock key_block(size);
  for (auto& pool : pools_) {
    auto it = pool.lower_bound(&key_block);
    if (it != pool.end()) {
      auto ptr = *it;

      // erase the block from pools
      pool.erase(it);
      return ptr;
    }
  }

  // try refill pools_[0]
  auto block = RefillPool(size);

  return block;
}

MemoryBlock* BuddyAllocator::SplitToAlloc(MemoryBlock* block, size_t size) {
  // pool_.erase(it);
  ptr_to_block_.insert({block->get_data(), block});

  VLOG(10) << "Split block (" << block->get_data() << ", " << block->get_size()
           << ") into";
  // split_flag == false means size equal to the block size
  // so no split action happened in the Split function
  bool split_flag = block->Split(size, &mb_pool_);

  VLOG(10) << "Left block (" << block->get_data() << ", " << block->get_size()
           << ")";
  block->set_type(MemoryBlock::ARENA_CHUNK);

  // the rest of memory if exist
  // if split_flag is true, then a new block is created inside Split function
  MemoryBlock* right_buddy = block->get_right_buddy();
  if (split_flag && right_buddy &&
      right_buddy->get_type() == MemoryBlock::FREE_CHUNK) {
    VLOG(10) << "Insert right block (" << right_buddy->get_data() << ", "
             << right_buddy->get_size() << ")";

    pools_[right_buddy->get_index()].insert(right_buddy);
  }
  return block;
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
