// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_map>

#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"

namespace paddle {
namespace memory {
namespace allocation {

using BlockList = std::list<BlockV2>;
using BlockListIt = BlockList::iterator;
using PtrBlockMap = std::unordered_map<void*, BlockListIt>;

class VMMAutoGrowthBestFitAllocatorV2 : public Allocator {
 public:
  VMMAutoGrowthBestFitAllocatorV2(
      const std::shared_ptr<CUDAVirtualMemAllocatorV2>& underlying_allocator,
      size_t alignment,
      const GPUPlace& place,
      PoolType pool_type);

  bool IsAllocThreadSafe() const override { return true; }

  const BlockList& all_blocks() const { return all_blocks_; }
  PoolType pool_type() const { return pool_type_; }
  size_t alignment() const { return alignment_; }

  bool SetBlockRemapEvent(void* ptr,
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
                          gpuStream_t stream,
                          gpuEvent_t event
#else
                          void* stream,
                          void* event
#endif
  );

 protected:
  phi::Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(phi::Allocation* allocation) override;

 private:
  phi::Allocation* AllocFromFreeBlocks(size_t size);
  void InsertFreeBlock(BlockListIt it);
  void EraseFreeBlock(BlockListIt it);
  void TryMerge(BlockListIt it);

  // Best-fit V2 only grows from the fixed-handle CUDA VMM provider. This
  // keeps the layer boundary explicit: the bottom allocator owns allocation
  // HandleLayout, while best-fit owns block-level BlockPartV2 state.
  std::shared_ptr<CUDAVirtualMemAllocatorV2> underlying_allocator_;
  size_t alignment_;
  GPUPlace place_;
  PoolType pool_type_;
  std::list<DecoratedAllocationPtr> underlying_allocations_;
  // Full block list ordered by VA address. This is the source of truth and
  // contains ACTIVE/FREE/GAP blocks together.
  BlockList all_blocks_;
  PtrBlockMap allocated_blocks_;
  std::map<std::pair<size_t, void*>, BlockListIt> free_blocks_;
  SpinLock spinlock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
