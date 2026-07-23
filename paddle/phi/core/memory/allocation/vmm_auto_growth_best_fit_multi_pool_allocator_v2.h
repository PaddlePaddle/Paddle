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

#if defined(PADDLE_WITH_CUDA)

#include <memory>

#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.h"
#include "paddle/phi/core/memory/mem_visitor.h"

namespace paddle {
namespace memory {
namespace allocation {

class VMMAutoGrowthBestFitMultiPoolAllocatorV2 : public Allocator {
 public:
  VMMAutoGrowthBestFitMultiPoolAllocatorV2(
      const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>& small_allocator,
      const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>& large_allocator,
      size_t small_allocation_threshold,
      const GPUPlace& place);

  bool IsAllocThreadSafe() const override { return true; }
  void Accept(AllocatorVisitor* visitor) override { visitor->Visit(this); }

  // Query free-block stats. When alloc_size > 0, returns stats for the
  // target pool only (the pool that would serve an allocation of that size).
  // When alloc_size == 0, aggregates across both pools (legacy behavior).
  void GetFreeBlockStats(size_t* total_free,
                         size_t* max_free,
                         size_t alloc_size = 0);

  bool SetBlockRemapEvent(void* ptr,
                          gpuStream_t stream,
                          std::shared_ptr<CUDAEventGuard> event);

  // Compacts the large VMM pool for a failed allocation request and optionally
  // reports whether remap was attempted. A zero request performs explicit
  // unbounded maintenance compaction.
  size_t RemapForAllocation(const Place& place,
                            size_t requested_size,
                            const VMMGrowOOMInfo* grow_oom = nullptr,
                            VMMRemapAttemptResult* attempt_result = nullptr);

  const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>& small_allocator()
      const {
    return small_allocator_;
  }
  const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>& large_allocator()
      const {
    return large_allocator_;
  }

 protected:
  phi::Allocation* AllocateImpl(size_t size) override;
  size_t CompactImpl(const Place& place) override;
  void FreeImpl(phi::Allocation* allocation) override;
  uint64_t ReleaseImpl(const Place& place) override;

 private:
  struct AllocationRoute {
    PoolType pool_type;
    VMMAutoGrowthBestFitAllocatorV2* allocator;
  };

  // Requests are split into small/large pools by a fixed threshold.
  AllocationRoute RouteAllocation(size_t size) const;

  std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2> small_allocator_;
  std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2> large_allocator_;
  size_t small_allocation_threshold_;
  GPUPlace place_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
