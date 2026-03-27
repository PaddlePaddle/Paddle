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

#include <mutex>
#include <unordered_map>
#include <vector>

#include "paddle/phi/backends/dynload/cuda_driver.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"

namespace paddle {
namespace memory {
namespace allocation {

// Compared with CUDAVirtualMemAllocator, V2 does not expose a single
// VA<->handle mapping per allocation. Instead it returns a lightweight
// HandleLayout (a handle list) for one allocation. Upper layers later
// transform that list into block-level BlockPartV2 state.
class CUDAVirtualMemAllocatorV2 : public Allocator {
 public:
  // Standalone use defaults to the transient pool. Upper layers may still
  // override this explicitly when routing by lifecycle.
  CUDAVirtualMemAllocatorV2(const GPUPlace& place,
                            size_t handle_size,
                            PoolType pool = PoolType::kTransient);

  bool IsAllocThreadSafe() const override;

  size_t handle_size() const { return handle_size_; }
  PoolType pool_type() const { return pool_type_; }
  VmmDevicePtr virtual_mem_base() const { return virtual_mem_base_; }
  size_t virtual_mem_size() const { return virtual_mem_size_; }
  size_t tail_offset() const { return virtual_mem_alloced_offset_; }
  // Best-fit/remap layers may consume VA from the reserved range incrementally.
  // V2 keeps this as an explicit cursor instead of reusing V1's
  // virtual_2_physical_map_ bookkeeping.
  void AdvanceTailOffset(size_t bytes) { virtual_mem_alloced_offset_ += bytes; }

  void UnmapHandle(VmmDevicePtr ptr, size_t size);
  void MapHandlesToVA(VmmDevicePtr ptr, const std::vector<VmmAllocHandle>& hs);
  // Exposes the allocation-level handle list for IPC/export queries. The key
  // is the raw allocation ptr returned by this allocator.
  bool CollectAllocationHandleLayout(void* ptr, HandleLayout* layout) const;

 protected:
  phi::Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(phi::Allocation* allocation) override;

 private:
  void InitOnce();
  void RegisterHandleLayout(void* ptr, const HandleLayout& layout);
  void UnregisterHandleLayout(void* ptr);

  GPUPlace place_;
  size_t handle_size_;
  PoolType pool_type_;
  std::once_flag init_flag_;

  VmmDevicePtr virtual_mem_base_{0};
  size_t virtual_mem_size_{0};
  size_t virtual_mem_alloced_offset_{0};
  size_t granularity_{0};
  CUmemAllocationProp prop_{};
  std::vector<CUmemAccessDesc> access_desc_;

  mutable std::unordered_map<void*, HandleLayout> allocation_layout_map_;
  mutable std::mutex allocation_layout_mu_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
