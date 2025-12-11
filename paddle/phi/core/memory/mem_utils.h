// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <list>
#include <map>
#include <set>
#include <vector>

#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/memory/allocation/vmm_ipc_allocation.h"

namespace paddle {

namespace memory {
namespace allocation {

struct BlockAllocation;
struct Block {
  Block(void* ptr,
        size_t size,
        bool is_free,
        BlockAllocation* allocation = nullptr)
      : ptr_(ptr), size_(size), is_free_(is_free), allocation_(allocation) {}

  void* ptr_;
  size_t size_;
  bool is_free_;
  BlockAllocation* allocation_;
  std::vector<BlockPart> parts_;
};

struct BlockAllocation : public Allocation {
  explicit BlockAllocation(const std::list<Block>::iterator& it,
                           phi::Place place)
      : Allocation(it->ptr_, it->size_, place), block_it_(it) {
    it->allocation_ = this;
  }
  std::list<Block>::iterator block_it_;
};
}  // namespace allocation

using allocation::Block;
using allocation::BlockAllocation;
/*!
 * Author: liujinnan
 * Note: MemoryCompactionStrategy is an abstract class that defines the
 * interface for memory compaction strategies. All memory compaction strategies
 * should inherit this base class and implement the corresponding interface.
 * Currently only supports the `TotalMemoryCompactor` strategy.
 */
class MemoryCompactionStrategy {
 public:
  /*!
   * \brief TryFuse will create new IterMark and returns an aggregated IterSum
   * that only has one IterSplit with the new IterMark.
   * \param blocks A list of memory blocks to be compacted.
   * \param start_ptr A pointer to the start of the memory blocks.
   * \param end_ptr A pointer to the end of the memory blocks.
   * \return if return -1 mean compact failed, else return compacted size.
   */
  virtual size_t Compact(std::list<Block>& blocks,  // NOLINT
                         void* start_ptr,
                         void* end_ptr) = 0;
};

// `TotalMemoryCompactor` strategy will compact all free blocks to the
// whole memory pool by moving the non-free blocks.
class TotalMemoryCompactor final : public MemoryCompactionStrategy {
 public:
  size_t Compact(std::list<Block>& blocks,  // NOLINT
                 void* start_ptr,
                 void* end_ptr) override;
};

#if defined(PADDLE_WITH_CUDA)
// return a pair of <largest_free_block_size, sum_of_n_largest_free_block_size>
PADDLE_API extern std::pair<size_t, size_t> VmmMaxFreeSize(
    const phi::GPUPlace& place, int32_t n);

// Try using Allocator to simulate an allocation, simulating a request for
// vector<size>.
PADDLE_API extern bool TryAllocBatch(const phi::GPUPlace& place,
                                     const std::vector<size_t>& sizes);

// Compact memory of free blocks held by the VmmAllocator.
PADDLE_API extern size_t VmmCompact(const phi::GPUPlace& place);

// Get VMM allocator free block info.
PADDLE_API extern std::vector<std::vector<std::pair<size_t, uintptr_t>>>
FreeBlockInfoOfVmmAllocator(const phi::GPUPlace& place);

// Get VMM allocator all block info.
PADDLE_API extern std::vector<std::vector<std::tuple<size_t, uintptr_t, bool>>>
AllBlockInfoOfVmmAllocator(const phi::GPUPlace& place);

// Get allocate event when start FLAGS_record_alloc_event.
PADDLE_API extern std::vector<
    std::tuple<uintptr_t, bool, uint64_t, size_t, int64_t, int64_t>>
GetAllocateEvent(const phi::GPUPlace& place);

// Get compact count and size when start FLAGS_enable_compact_mem.
PADDLE_API extern std::vector<size_t> GetCompactSize(
    const phi::GPUPlace& place);
#endif

}  // namespace memory
}  // namespace paddle
