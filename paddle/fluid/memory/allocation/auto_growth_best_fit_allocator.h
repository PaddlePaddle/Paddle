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

#pragma once

#include <list>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <utility>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

class AutoGrowthBestFitAllocator : public Allocator {
 public:
  explicit AutoGrowthBestFitAllocator(
      const std::shared_ptr<Allocator> &underlying_allocator, size_t chunk_size,
      size_t alignment);

  bool IsAllocThreadSafe() const override { return true; }

  using AllocationList = std::list<AllocationPtr>;
  using AllocationListIt = AllocationList::iterator;

  struct Chunk {
    struct Block {
      Block(void *ptr, size_t size, bool is_free, Chunk *chunk)
          : ptr_(ptr), size_(size), is_free_(is_free), chunk_(chunk) {}

      void *ptr_;
      size_t size_;
      bool is_free_;
      Chunk *chunk_;  // which chunk it is from
    };

    explicit Chunk(AllocationPtr allocation)
        : allocation_(std::move(allocation)) {}

    AllocationPtr allocation_;
    std::list<Block> blocks_;
    // std::mutex mtx_;

    struct BlockAllocation : public Allocation {
      explicit BlockAllocation(const std::list<Block>::iterator &it)
          : Allocation(it->ptr_, it->size_, it->chunk_->allocation_->place()),
            block_it_(it) {}

      std::list<Block>::iterator block_it_;
    };
  };

 protected:
  Allocation *AllocateImpl(size_t size, Attr attr) override;

  void FreeImpl(Allocation *allocation) override;

 private:
  using BlockIt = std::list<Chunk::Block>::iterator;

  std::shared_ptr<Allocator> underlying_allocator_;
  std::list<Chunk> chunks_;
  std::map<std::pair<size_t, void *>, BlockIt> free_blocks_;
  size_t chunk_size_;
  size_t alignment_;

  bool underlying_allocator_exhaustive_{false};

  mutable std::mutex mtx_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
