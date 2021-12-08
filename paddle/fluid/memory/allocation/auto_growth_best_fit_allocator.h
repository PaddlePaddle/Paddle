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

#pragma once

#include <list>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <utility>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/spin_lock.h"

namespace paddle {
namespace memory {
namespace allocation {

class AutoGrowthBestFitAllocator : public Allocator {
 public:
  AutoGrowthBestFitAllocator(
      const std::shared_ptr<Allocator> &underlying_allocator, size_t alignment,
      size_t chunk_size = 0, bool allow_free_idle_chunk = true);

  bool IsAllocThreadSafe() const override { return true; }

 protected:
  Allocation *AllocateImpl(size_t size) override;

  void FreeImpl(Allocation *allocation) override;

  // Release the memory block which is not used in pool.
  uint64_t ReleaseImpl(const platform::Place &place) override {
    return FreeIdleChunks();
  }

 private:
  uint64_t FreeIdleChunks();

  template <typename T>
  using List = std::list<T>;

  struct Chunk;

  struct Block {
    Block(void *ptr, size_t size, bool is_free, Chunk *chunk)
        : ptr_(ptr), size_(size), is_free_(is_free), chunk_(chunk) {}

    void *ptr_;
    size_t size_;
    bool is_free_;
    Chunk *chunk_;  // which chunk it is from
  };

  struct Chunk {
    explicit Chunk(AllocationPtr allocation)
        : allocation_(std::move(allocation)) {}

    AllocationPtr allocation_;
    List<Block> blocks_;
  };

  struct BlockAllocation : public Allocation {
    explicit BlockAllocation(const List<Block>::iterator &it)
        : Allocation(it->ptr_, it->chunk_->allocation_->base_ptr(), it->size_,
                     it->chunk_->allocation_->place()),
          block_it_(it) {}

    List<Block>::iterator block_it_;
  };

  using BlockIt = List<Block>::iterator;

  std::shared_ptr<Allocator> underlying_allocator_;
  std::map<std::pair<size_t, void *>, BlockIt> free_blocks_;
  std::list<Chunk> chunks_;
  size_t alignment_;
  size_t chunk_size_;
  bool allow_free_idle_chunk_;

  SpinLock spinlock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
