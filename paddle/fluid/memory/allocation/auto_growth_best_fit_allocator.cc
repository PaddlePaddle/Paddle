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

#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator.h"
#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>

namespace paddle {
namespace memory {
namespace allocation {

static size_t align(size_t size, size_t alignment) {
  auto remaining = size % alignment;
  return remaining == 0 ? size : size + alignment - remaining;
}

AutoGrowthBestFitAllocator::AutoGrowthBestFitAllocator(
    const std::shared_ptr<Allocator> &underlying_allocator, size_t chunk_size,
    size_t alignment)
    : underlying_allocator_(underlying_allocator),
      chunk_size_(align(chunk_size, alignment)),
      alignment_(alignment) {}

Allocation *AutoGrowthBestFitAllocator::AllocateImpl(size_t size, Attr attr) {
  size = align(size, alignment_);
  std::lock_guard<std::mutex> guard(mtx_);
  auto iter = free_blocks_.lower_bound(std::make_pair(size, nullptr));
  BlockIt block_it;
  if (iter != free_blocks_.end()) {
    VLOG(2) << "Found " << iter->second->size_ << " for " << size;
    block_it = iter->second;
    free_blocks_.erase(iter);
    auto *chunk = block_it->chunk_;
    size_t remaining_size = block_it->size_ - size;
    if (remaining_size == 0) {
      block_it->is_free_ = false;
      VLOG(2) << "Found and no remaining";
    } else {
      auto remaining_free_block = chunk->blocks_.insert(
          block_it, Chunk::Block(block_it->ptr_, remaining_size, true, chunk));
      free_blocks_.emplace(std::make_pair(remaining_size, block_it->ptr_),
                           remaining_free_block);
      block_it->ptr_ =
          reinterpret_cast<uint8_t *>(block_it->ptr_) + remaining_size;
      block_it->size_ = size;
      block_it->is_free_ = false;
      VLOG(2) << "Found and remaining " << remaining_size;
    }
  } else {
    size_t alloc_size = size;
    if (!underlying_allocator_exhaustive_ && chunk_size_ > size) {
      alloc_size = chunk_size_;
    }

    try {
      chunks_.emplace_back(underlying_allocator_->Allocate(alloc_size, attr));
    } catch (BadAlloc &ex) {
      if (size == alloc_size) throw ex;
      underlying_allocator_exhaustive_ = true;
      alloc_size = size;
      chunks_.emplace_back(underlying_allocator_->Allocate(alloc_size, attr));
    }
    auto *chunk = &(*chunks_.rbegin());
    uint8_t *p = reinterpret_cast<uint8_t *>(chunk->allocation_->ptr());
    auto &blocks = chunk->blocks_;

    size_t remaining_size = alloc_size - size;
    if (remaining_size > 0) {
      blocks.emplace_back(p, remaining_size, true, chunk);
      free_blocks_.emplace(std::make_pair(remaining_size, p), --(blocks.end()));
    }
    blocks.emplace_back(p + remaining_size, size, false, chunk);
    block_it = --(blocks.end());
    VLOG(2) << "Not found and allocate " << alloc_size << ", and remaining "
            << remaining_size;
  }
  VLOG(2) << "After allocate, free blocks " << free_blocks_.size();
  return new Chunk::BlockAllocation(block_it);
}

void AutoGrowthBestFitAllocator::FreeImpl(Allocation *allocation) {
  auto &block_it = static_cast<Chunk::BlockAllocation *>(allocation)->block_it_;
  auto &blocks = block_it->chunk_->blocks_;

  std::lock_guard<std::mutex> guard(mtx_);
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

  VLOG(2) << "Combine " << block_it->size_ << ", " << blocks.size() << ", "
          << free_blocks_.size();
  delete allocation;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
