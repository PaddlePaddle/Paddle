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

#pragma once

#include <list>
#include <map>
#include <set>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/spin_lock.h"

namespace paddle {
namespace memory {
namespace allocation {

struct Block {
  Block(void *ptr, size_t size, bool is_free)
      : ptr_(ptr), size_(size), is_free_(is_free) {}

  void *ptr_;
  size_t size_;
  bool is_free_;
};

struct BlockAllocation : public Allocation {
  explicit BlockAllocation(const std::list<Block>::iterator &it,
                           platform::Place place)
      : Allocation(it->ptr_, it->size_, place), block_it_(it) {}

  std::list<Block>::iterator block_it_;
};

/**
 * Like AutoGrowthBestFitAllocator, VirtualMemoryAutoGrowthBestFitAllocator will
 * gradually apply to GPU for video memory as the model uses more video memory.
 * However, the difference is that VirtualMemoryAutoGrowthBestFitAllocator uses
 * nviaid's virtual memory management technology and obtains the virtual memory
 * address. If the video memory applied for twice is continuous, we can combine
 * the two video memories later. This combination can greatly reduce
 * fragmentation.
 */
class VirtualMemoryAutoGrowthBestFitAllocator : public Allocator {
 public:
  VirtualMemoryAutoGrowthBestFitAllocator(
      const std::shared_ptr<Allocator> &underlying_allocator, size_t alignment,
      const platform::CUDAPlace &place);

  bool IsAllocThreadSafe() const override { return true; }

 protected:
  phi::Allocation *AllocateImpl(size_t size) override;

  void FreeImpl(phi::Allocation *allocation) override;

 private:
  phi::Allocation *AllocFromFreeBlocks(size_t size);
  void ExtendAndMerge(size_t size);
  void TryMergeBlock2Blocks(std::list<Block>::iterator iter);

  std::shared_ptr<Allocator> underlying_allocator_;
  size_t alignment_;

  std::map<std::pair<size_t, void *>, std::list<Block>::iterator> free_blocks_;
  std::list<Block> all_blocks_;
  std::list<AllocationPtr> allocations_;
  platform::Place place_;
  SpinLock spinlock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
