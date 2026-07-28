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
#include "paddle/phi/core/memory/allocation/vmm_ipc_allocation.h"

namespace paddle {
namespace memory {
namespace allocation {

struct VMMGrowOOMInfo {
  size_t requested_handles{0};
  size_t created_handles{0};
  size_t handle_size{0};
  int device{-1};
  PoolType pool_type{PoolType::kLarge};
};

// Carries the exact handle-creation progress from a failed VMM allocation.
// OOM recovery can use this immutable per-request snapshot without consulting
// shared "last failure" state.
class VMMGrowOOM : public BadAlloc {
 public:
  VMMGrowOOM(std::string message,
             const char* file,
             int line,
             VMMGrowOOMInfo info)
      : BadAlloc(std::move(message), file, line), info_(info) {}

  const VMMGrowOOMInfo& info() const { return info_; }

 private:
  VMMGrowOOMInfo info_;
};

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

  struct StagedRemapDestination {
    Allocation* allocation{nullptr};
    BlockV2 block;
  };

  struct ReleaseDriverStats {
    uint64_t allocation_count{0};
    uint64_t handle_count{0};
    uint64_t released_bytes{0};
    uint64_t skipped_owned_handles{0};
    uint64_t unmap_calls{0};
    uint64_t unmap_us{0};
    uint64_t release_calls{0};
    uint64_t release_us{0};
    uint64_t metadata_us{0};
  };

  struct AllocationLayoutRegistry {
    void Add(Allocation* allocation, void* ptr, const HandleLayout& layout);
    bool Lookup(void* ptr, HandleLayout* layout) const;
    bool Lookup(Allocation* allocation, HandleLayout* layout) const;
    void Remove(Allocation* allocation, void* ptr);

   private:
    struct PtrEntry {
      Allocation* allocation{nullptr};
      HandleLayout layout;
    };
    std::unordered_map<void*, PtrEntry> layouts_by_ptr_;
    std::unordered_map<Allocation*, HandleLayout> layouts_by_allocation_;
    mutable SpinLock spinlock_;
  };

  // Standalone use defaults to the large pool. Upper layers may also choose
  // explicit small/large pool types.
  CUDAVirtualMemAllocatorV2(const GPUPlace& place,
                            size_t handle_size,
                            PoolType pool = PoolType::kLarge);
  ~CUDAVirtualMemAllocatorV2() override;

  bool IsAllocThreadSafe() const override;

  size_t handle_size() const { return handle_size_; }
  PoolType pool_type() const { return pool_type_; }
  VMMDevicePtr virtual_mem_base() const { return virtual_mem_base_; }
  size_t virtual_mem_size() const { return virtual_mem_size_; }
  size_t tail_offset() const { return virtual_mem_alloced_offset_; }
  ReleaseDriverStats GetReleaseDriverStats() const {
    return release_driver_stats_;
  }
  // Best-fit/remap layers may consume VA from the reserved range incrementally.
  // V2 keeps this as an explicit cursor instead of reusing V1's
  // virtual_2_physical_map_ bookkeeping.
  void AdvanceTailOffset(size_t bytes) { virtual_mem_alloced_offset_ += bytes; }
  // Retreat the tail cursor when the compactor discovers that blocks no
  // longer span up to the previous high-water mark (e.g. after
  // FreeIdleChunks released tail-end underlying allocations).
  void SetTailOffset(size_t offset) { virtual_mem_alloced_offset_ = offset; }

  void RollbackMappedHandleRange(VMMDevicePtr ptr, size_t handle_count);
  struct MoveBackingPageStats {
    uint64_t unmap_us{0};
    uint64_t map_us{0};
    uint64_t set_access_us{0};
    uint64_t metadata_us{0};
    uint64_t restore_us{0};
    uint64_t rollback_us{0};
    uint64_t unmap_calls{0};
    uint64_t set_access_calls{0};
  };
  bool UnmapMappedRangeForRemap(VMMDevicePtr ptr,
                                size_t handle_count,
                                MoveBackingPageStats* stats = nullptr);
  bool MoveBackingPage(const VMMBackingMap::MappedPage& source,
                       const VMMBackingMap::UnmappedPage& target,
                       MoveBackingPageStats* stats = nullptr,
                       bool defer_target_access = false,
                       bool source_already_unmapped = false);
  bool MoveBackingPageForRemap(const VMMBackingMap::MappedPage& source,
                               const VMMBackingMap::UnmappedPage& target,
                               const std::shared_ptr<VMMHandleMeta>& meta,
                               MoveBackingPageStats* stats = nullptr,
                               bool defer_target_access = false,
                               bool source_already_unmapped = false);
  bool SetAccessForMappedRange(VMMDevicePtr ptr,
                               size_t size,
                               MoveBackingPageStats* stats = nullptr);
  enum class RestoreRemapSourceResult : uint8_t {
    kSkipped = 0,
    kRestored = 1,
    kForceReleased = 2,
  };
  RestoreRemapSourceResult RestoreRemapSourceMapping(
      VMMAllocHandle handle,
      const std::shared_ptr<VMMHandleMeta>& meta,
      size_t size);
  RestoreRemapSourceResult ForceReleaseRemapSource(
      VMMAllocHandle handle,
      const std::shared_ptr<VMMHandleMeta>& meta,
      size_t size,
      const char* context,
      bool unmap_mapped_source);

  const GPUPlace& place() const { return place_; }
  AllocationWithBlock AppendWithBlock(size_t size);
  // Create fresh physical backing and map it at an existing reserved VA range.
  // This is used by upper layers to reuse unmapped-free VA space in place.
  AllocationWithBlock PlaceAtVAWithBlock(VMMDevicePtr ptr, size_t size);
  bool IsRemapDestinationAllocation(void* ptr) const;
  bool ClearRemapDestinationOwnership(VMMDevicePtr ptr, size_t size);
  // Clears ownership only for backing pages fully covered by [ptr, ptr + size).
  size_t ClearRemapDestinationOwnershipInRange(VMMDevicePtr ptr, size_t size);

  // Create a staged allocation and mapped-free block for handles moved by
  // remap compaction. The handles already exist (cuMemCreate was done earlier);
  // rollback paths must destroy the staged allocation before discarding the
  // block view.
  StagedRemapDestination CreateStagedRemapDestination(
      VMMDevicePtr ptr,
      const std::vector<VMMBackingMap::MappedPage>& source_pages,
      size_t start,
      size_t count,
      PoolType pool_type);
  DecoratedAllocationPtr AdoptRemapDestinationAllocation(
      Allocation* allocation);
  void DestroyStagedDestinationAllocation(Allocation* allocation);

  bool HasIPCExportedRange(VMMDevicePtr ptr, size_t size) const;
  size_t CountIPCExportedBytes(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const;
  bool IsRangeUnmapped(VMMDevicePtr ptr, size_t size) const;
  bool IsRangeReleasable(VMMDevicePtr ptr, size_t size) const;
  // Metadata-only lookup. This does not pin backing or create export FDs.
  bool CollectIPCParts(VMMDevicePtr ptr,
                       size_t size,
                       std::vector<BlockPart>* ipc_parts) const;
  // Validate and pin the backing before creating reusable export FDs.
  bool ExportIPCParts(VMMDevicePtr ptr,
                      size_t size,
                      std::vector<BlockPart>* ipc_parts);
  bool SetBlockRemapEvent(const BlockV2& block,
                          gpuStream_t stream,
                          std::shared_ptr<CUDAEventGuard> event);
  std::vector<VMMBackingMap::MappedPage> CollectMappedPages(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
      size_t target_bytes) const;
  std::vector<VMMBackingMap::MappedPage> CollectRemapSourcePages(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
      size_t target_bytes) const;
  std::vector<VMMBackingMap::UnmappedPage> CollectUnmappedPages(
      const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
      size_t target_bytes) const;
  bool ValidateMappedPages(const std::vector<VMMBackingMap::MappedPage>& pages,
                           const char* context) const;
  bool ValidateUnmappedPages(
      const std::vector<VMMBackingMap::UnmappedPage>& pages,
      const char* context) const;

 protected:
  phi::Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(phi::Allocation* allocation) override;

 private:
  void InitOnce();
  bool IsReservedVARange(VMMDevicePtr ptr, size_t size) const;
  void BuildIPCParts(const std::vector<IPCPartDescriptor>& descriptors,
                     bool include_shared_fd,
                     std::vector<BlockPart>* ipc_parts) const;
  int GetOrCreateIPCExportFD(VMMAllocHandle handle) const;
  void CloseIPCExportFD(VMMAllocHandle handle) const;
  bool SetRemapEvent(VMMDevicePtr ptr,
                     size_t size,
                     gpuStream_t stream,
                     std::shared_ptr<CUDAEventGuard> event);
  void RollbackCreatedHandles(const HandleLayout& layout) const;
  void MarkLayoutMapped(const HandleLayout& layout);
  void MarkRemapDestinationLayoutMapped(const HandleLayout& layout);
  AllocationWithLayout AppendWithLayout(size_t size);
  AllocationWithLayout PlaceAtVAWithLayout(VMMDevicePtr ptr, size_t size);
  HandleLayout CreateMappedHandleLayout(VMMDevicePtr ptr,
                                        size_t aligned_size,
                                        const char* context,
                                        bool is_grow = false);
  void SetAccessOrThrow(VMMDevicePtr ptr,
                        size_t aligned_size,
                        size_t num_handles,
                        const char* context);
  bool CollectAllocationHandleLayout(void* ptr, HandleLayout* layout) const;
  bool IsRemapDestinationOwnedLayout(const HandleLayout& layout) const;
  AllocationWithLayout WrapTrackedAllocation(VMMDevicePtr ptr,
                                             size_t size,
                                             HandleLayout layout,
                                             bool advance_tail);
  AllocationWithBlock BuildAllocationWithBlock(
      AllocationWithLayout allocation_with_layout);
  Allocation* CreateTrackedAllocation(VMMDevicePtr ptr,
                                      size_t size,
                                      const HandleLayout& layout);
  void RegisterHandleLayout(Allocation* allocation,
                            void* ptr,
                            const HandleLayout& layout);
  HandleLayout RequireHandleLayout(Allocation* allocation) const;
  void UnregisterHandleLayout(Allocation* allocation, void* ptr);

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
  ReleaseDriverStats release_driver_stats_;
  mutable SpinLock ipc_export_lock_;
  mutable std::unordered_map<VMMAllocHandle, int> ipc_export_fds_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
