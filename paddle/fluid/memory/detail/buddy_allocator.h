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

#pragma once

#include <array>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/memory/detail/memory_block.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/gpu_info.h"

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"

namespace paddle {
namespace memory {
namespace detail {

class BuddyAllocator {
 public:
  BuddyAllocator();
  BuddyAllocator(std::unique_ptr<SystemAllocator> system_allocator,
                 size_t min_chunk_size, size_t max_chunk_size);
  ~BuddyAllocator();

 public:
  void Init(std::unique_ptr<SystemAllocator> system_allocator,
            size_t min_chunk_size, size_t max_chunk_size);
  void* Alloc(size_t unaligned_size);
  void Free(void* ptr);
  size_t Used();
  size_t GetMinChunkSize();
  size_t GetMaxChunkSize();

 public:
  // Disable copy and assignment
  BuddyAllocator(const BuddyAllocator&) = delete;
  BuddyAllocator& operator=(const BuddyAllocator&) = delete;

 private:
  // Each element in PoolSet is a free allocation
  // using Pool = std::set<MemoryBlock*, MemoryBlockComparator>;
  using Pool = absl::btree_set<MemoryBlock*, MemoryBlockComparator>;

  /*! \brief Allocate fixed-size memory from system */
  void* SystemAlloc(size_t size);

  /*! \brief If existing chunks are not suitable, refill pool */
  MemoryBlock* RefillPool(size_t request_bytes);

  /**
   *  \brief   Find the suitable chunk from existing pool and split
   *           it to left and right buddies
   *
   *  \param   it     the iterator of pool list
   *  \param   size   the size of allocation
   *
   *  \return  the left buddy address
   */
  MemoryBlock* SplitToAlloc(MemoryBlock*, size_t size);

  /*! \brief Find the existing chunk which used to allocation */
  // Pool::iterator FindExistChunk(size_t size);
  MemoryBlock* FindExistChunk(size_t size);

 private:
  size_t total_used_ = 0;  // the total size of used memory
  size_t total_free_ = 0;  // the total size of free memory

  size_t min_chunk_size_;  // the minimum size of each chunk
  size_t max_chunk_size_;  // the maximum size of each chunk

  size_t realloc_size_ = 0;  // the size of re-allocated chunk

 private:
  /**
   * \brief A list of free allocation
   *
   * \note  Only store free chunk memory in pool
   */
  std::array<Pool, 2> pools_;

 private:
  // std::unordered_map<void*, MemoryBlock*> ptr_to_block_;
  absl::flat_hash_map<void*, MemoryBlock*> ptr_to_block_;
  MemoryBlockPool mb_pool_;

 private:
  /*! Allocate CPU/GPU memory from system */
  std::unique_ptr<SystemAllocator> system_allocator_;
  std::mutex mutex_;
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
