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

#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"

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
  BlockAllocation* allocation_;
  size_t size_;
  bool is_free_;
};

struct BlockAllocation : public Allocation {
  explicit BlockAllocation(const std::list<Block>::iterator& it,
                           phi::Place place)
      : Allocation(it->ptr_, it->size_, place), block_it_(it) {
    it->allocation_ = this;
  }
  ~BlockAllocation() override {
    if (block_it_ != std::list<Block>::iterator{}) {
      if (block_it_->allocation_) block_it_->allocation_ = nullptr;
      block_it_ = std::list<Block>::iterator{};
    }
  }
  std::list<Block>::iterator block_it_;
};

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
   * \return Whether the defragmentation was successful.
   */
  virtual bool compact(std::list<Block>& blocks,  // NOLINT
                       void* start_ptr,
                       void* end_ptr) = 0;
};

// `TotalMemoryCompactor` strategy will compact all free blocks to the
// whole memory pool by moving the non-free blocks.
class TotalMemoryCompactor final : public MemoryCompactionStrategy {
 public:
  bool compact(std::list<Block>& blocks,  // NOLINT
               void* start_ptr,
               void* end_ptr) override;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
