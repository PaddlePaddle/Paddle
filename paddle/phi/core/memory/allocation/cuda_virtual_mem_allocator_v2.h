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

#include <unordered_map>
#include <vector>

#include "paddle/phi/backends/dynload/cuda_driver.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"
#include "paddle/phi/core/memory/allocation/vmm_backing_map.h"

namespace paddle {
namespace memory {
namespace allocation {

// Compared with CUDAVirtualMemAllocator, V2 does not expose a single
// VA<->handle mapping per allocation. It keeps the handle layout registered in
// the bottom allocator and hands upper layers either allocation-level layout
// snapshots or materialized mapped-free BlockV2 views.
class CUDAVirtualMemAllocatorV2 : public Allocator {
 public:
  struct AllocationWithLayout {
    DecoratedAllocationPtr allocation;
    HandleLayout layout;
  };

  struct AllocationWithBlock {
    bool HasAllocation() const { return allocation != nullptr; }
    BlockV2 TakeBlock() { return std::move(block); }
    DecoratedAllocationPtr TakeAllocation() { return std::move(allocation); }

    DecoratedAllocationPtr allocation;
    BlockV2 block;
  };

  struct AllocationLayoutRegistry {
    void Add(void* ptr, const HandleLayout& layout);
    bool Lookup(void* ptr, HandleLayout* layout) const;
    void Remove(void* ptr);

   private:
    std::unordered_map<void*, HandleLayout> layouts_;
    mutable SpinLock spinlock_;
  };

  // Standalone use defaults to the large pool. Upper layers may also choose
  // explicit small/large pool types.
  CUDAVirtualMemAllocatorV2(const GPUPlace& place,
                            size_t handle_size,
                            PoolType pool = PoolType::kLarge);

  bool IsAllocThreadSafe() const override;

  size_t handle_size() const { return handle_size_; }
  PoolType pool_type() const { return pool_type_; }
  VMMDevicePtr virtual_mem_base() const { return virtual_mem_base_; }
  size_t virtual_mem_size() const { return virtual_mem_size_; }
  size_t tail_offset() const { return virtual_mem_alloced_offset_; }
  // Best-fit layers may consume VA from the reserved range incrementally. V2
  // keeps this as an explicit cursor instead of reusing V1's
  // virtual_2_physical_map_ bookkeeping.
  void AdvanceTailOffset(size_t bytes) { virtual_mem_alloced_offset_ += bytes; }
  // Retreat the tail cursor after upper layers release tail-end backing.
  void SetTailOffset(size_t offset) { virtual_mem_alloced_offset_ = offset; }

  const GPUPlace& place() const { return place_; }
  AllocationWithBlock AppendWithBlock(size_t size);
  // Create fresh physical backing and map it at an existing reserved VA range.
  // This is used by upper layers to reuse unmapped-free VA space in place.
  AllocationWithBlock PlaceAtVAWithBlock(VMMDevicePtr ptr, size_t size);
  bool IsRangeReleasable(VMMDevicePtr ptr, size_t size) const;

 protected:
  phi::Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(phi::Allocation* allocation) override;

 private:
  void InitOnce();
  void RollbackCreatedHandles(const HandleLayout& layout) const;
  void MarkLayoutMapped(const HandleLayout& layout);
  AllocationWithLayout AppendWithLayout(size_t size);
  AllocationWithLayout PlaceAtVAWithLayout(VMMDevicePtr ptr, size_t size);
  HandleLayout CreateMappedHandleLayout(VMMDevicePtr ptr,
                                        size_t aligned_size,
                                        const char* context);
  void SetAccessOrThrow(VMMDevicePtr ptr,
                        size_t aligned_size,
                        size_t num_handles,
                        const char* context);
  bool CollectAllocationHandleLayout(void* ptr, HandleLayout* layout) const;
  AllocationWithLayout WrapTrackedAllocation(VMMDevicePtr ptr,
                                             size_t size,
                                             HandleLayout layout,
                                             bool advance_tail);
  AllocationWithBlock BuildAllocationWithBlock(
      AllocationWithLayout allocation_with_layout);
  Allocation* CreateTrackedAllocation(VMMDevicePtr ptr,
                                      size_t size,
                                      const HandleLayout& layout);
  void RegisterHandleLayout(void* ptr, const HandleLayout& layout);
  HandleLayout RequireHandleLayout(void* ptr) const;
  void UnregisterHandleLayout(void* ptr);

  GPUPlace place_;
  size_t handle_size_;
  PoolType pool_type_;
  std::once_flag init_flag_;

  VMMDevicePtr virtual_mem_base_{0};
  size_t virtual_mem_size_{0};
  size_t virtual_mem_alloced_offset_{0};
  size_t granularity_{0};
  CUmemAllocationProp prop_{};
  std::vector<CUmemAccessDesc> access_desc_;

  AllocationLayoutRegistry allocation_layouts_;
  VMMBackingMap backing_map_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
