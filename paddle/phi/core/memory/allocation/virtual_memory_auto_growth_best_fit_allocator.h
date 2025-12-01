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
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/memory/allocation/vmm_ipc_allocation.h"
#include "paddle/phi/core/memory/mem_utils.h"
#include "paddle/phi/core/memory/mem_visitor.h"

namespace paddle {
namespace memory {

namespace allocation {
/**
 * Like AutoGrowthBestFitAllocator, VirtualMemoryAutoGrowthBestFitAllocator will
 * gradually apply to GPU for video memory as the model uses more video memory.
 * However, the difference is that VirtualMemoryAutoGrowthBestFitAllocator uses
 * NVIDIA's virtual memory management technology and obtains the virtual memory
 * address. If the video memory applied for twice is continuous, we can combine
 * the two video memories later. This combination can greatly reduce
 * fragmentation.
 */

class VirtualMemoryAutoGrowthBestFitAllocator : public Allocator {
 public:
  VirtualMemoryAutoGrowthBestFitAllocator(
      const std::shared_ptr<Allocator> &underlying_allocator,
      size_t alignment,
      const phi::GPUPlace &place);

  std::shared_ptr<Allocator> &GetUnderLyingAllocator() {
    return underlying_allocator_;
  }
  const std::map<std::pair<size_t, void *>, std::list<Block>::iterator>
      &GetFreeBlocks() const {
    return free_blocks_;
  }

  const std::list<Block> &GetAllBlocks() const { return all_blocks_; }

  std::pair<size_t, size_t> SumLargestFreeBlockSizes(int32_t n) const;
  void Accept(AllocatorVisitor *visitor) override { visitor->Visit(this); }

  bool IsAllocThreadSafe() const override { return true; }
  void PreAlloc() override;
  void PreAllocate(size_t size);
  // Try to simulate an allocation, simulating a request for vector<size>.

  bool TryAllocateBatch(const std::vector<size_t> &sizes);

  bool CollectTensorParts(void *ptr, std::vector<BlockPart> *parts);

 protected:
  phi::Allocation *AllocateImpl(size_t size) override;
  size_t CompactImpl(const phi::Place &place) override;
  void FreeImpl(phi::Allocation *allocation) override;

 private:
  // AllocateOrCompact will try to allocate memory from free blocks first, if
  // OOM happens, it will try to compact memory.
  std::optional<AllocationPtr> AllocateOrCompact(size_t size);
  phi::Allocation *AllocFromFreeBlocks(size_t size);
  void ExtendOrCompact(size_t size);
  void TryMergeBlock2Blocks(std::list<Block>::iterator iter);
  void DumpInfo(std::string phase) const;

  std::shared_ptr<Allocator> underlying_allocator_;
  std::unique_ptr<MemoryCompactionStrategy> memory_compactor_;
  size_t alignment_;

  std::map<std::pair<size_t, void *>, std::list<Block>::iterator> free_blocks_;
  std::list<Block> all_blocks_;
  std::list<AllocationPtr> allocations_;
  phi::Place place_;
  SpinLock spinlock_;
};

/**
 * VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator is a multi-scale
 * allocator that combines the virtual memory management technology of
 * VirtualMemoryAutoGrowthBestFitAllocator and the multi-scale pooling strategy
 * of MultiScalePoolAllocator.
 */
class VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator
    : public MultiScalePoolAllocator {
 public:
  VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator(
      const std::shared_ptr<VirtualMemoryAutoGrowthBestFitAllocator>
          &small_allocator,
      const std::shared_ptr<VirtualMemoryAutoGrowthBestFitAllocator>
          &large_allocator,
      size_t alignment,
      const phi::GPUPlace &place)
      : MultiScalePoolAllocator(
            small_allocator, large_allocator, alignment, place) {}
  bool IsAllocThreadSafe() const override { return true; }
  void PreAlloc() override;
  void Accept(AllocatorVisitor *visitor) override { visitor->Visit(this); }
  bool IsSmallRequest(size_t size) override;
  std::vector<size_t> GetCompactSize() const { return compact_size_; }

 protected:
  size_t CompactImpl(const phi::Place &place) override;

 private:
  std::vector<size_t> compact_size_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
