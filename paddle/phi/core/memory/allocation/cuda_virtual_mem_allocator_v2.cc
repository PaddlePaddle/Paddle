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

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

#if defined(PADDLE_WITH_CUDA)

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

#if defined(__linux__)
#include <unistd.h>
#endif

#include "glog/logging.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/scope_guard.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

constexpr size_t kVMMSetAccessChunkSize = 64UL << 20;

using Clock = std::chrono::steady_clock;

uint64_t ElapsedMicros(Clock::time_point start, Clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());
}

bool IsCudaDeinitialized(CUresult result) {
  return result == CUDA_ERROR_DEINITIALIZED;
}

bool IsCudaRuntimeDeinitialized(cudaError_t result) {
  return result == cudaErrorCudartUnloading;
}

size_t GetPoolVAMultiplier(PoolType pool_type) {
  switch (pool_type) {
    case PoolType::kSmall:
      return 1;
    case PoolType::kLarge:
      return 4;
  }
  return 1;
}

struct SetAccessResult {
  CUresult status{CUDA_SUCCESS};
  size_t failed_offset{0};
  size_t failed_size{0};
  size_t call_count{0};
};

SetAccessResult SetAccessInChunks(VMMDevicePtr ptr,
                                  size_t size,
                                  size_t handle_size,
                                  const std::vector<CUmemAccessDesc>& desc) {
  const size_t chunk_size =
      std::max(handle_size, AlignedSize(kVMMSetAccessChunkSize, handle_size));
  SetAccessResult result;
  size_t offset = 0;
  while (offset < size) {
    const size_t remaining = size - offset;
    const size_t current_size = std::min(chunk_size, remaining);
    auto status = phi::dynload::cuMemSetAccess(
        ptr + offset, current_size, desc.data(), desc.size());
    ++result.call_count;
    if (status != CUDA_SUCCESS) {
      result.status = status;
      result.failed_offset = offset;
      result.failed_size = current_size;
      return result;
    }
    offset += current_size;
  }
  return result;
}

SetAccessResult SetAccessWholeRange(VMMDevicePtr ptr,
                                    size_t size,
                                    const std::vector<CUmemAccessDesc>& desc) {
  auto status =
      phi::dynload::cuMemSetAccess(ptr, size, desc.data(), desc.size());
  if (status != CUDA_SUCCESS) {
    return {status, 0, size, 1};
  }
  return {CUDA_SUCCESS, 0, 0, 1};
}

template <typename Map, typename Key, typename Value>
void EmplaceOrEnforce(Map* map,
                      Key&& key,
                      Value&& value,
                      const char* map_name) {
  const bool inserted =
      map->try_emplace(std::forward<Key>(key), std::forward<Value>(value))
          .second;
  PADDLE_ENFORCE_EQ(
      inserted,
      true,
      common::errors::AlreadyExists(
          "Duplicate key inserted into %s, allocator state is inconsistent.",
          map_name));
}

}  // namespace

void CUDAVirtualMemAllocatorV2::AllocationLayoutRegistry::Add(
    Allocation* allocation, void* ptr, const HandleLayout& layout) {
  std::lock_guard<SpinLock> guard(spinlock_);
  EmplaceOrEnforce(&layouts_by_allocation_,
                   allocation,
                   layout,
                   "allocation_layout_by_allocation_");
  layouts_by_ptr_[ptr] = PtrEntry{allocation, layout};
}

bool CUDAVirtualMemAllocatorV2::AllocationLayoutRegistry::Lookup(
    void* ptr, HandleLayout* layout) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto it = layouts_by_ptr_.find(ptr);
  if (it == layouts_by_ptr_.end()) {
    return false;
  }
  if (layout != nullptr) {
    *layout = it->second.layout;
  }
  return true;
}

bool CUDAVirtualMemAllocatorV2::AllocationLayoutRegistry::Lookup(
    Allocation* allocation, HandleLayout* layout) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto it = layouts_by_allocation_.find(allocation);
  if (it == layouts_by_allocation_.end()) {
    return false;
  }
  if (layout != nullptr) {
    *layout = it->second;
  }
  return true;
}

void CUDAVirtualMemAllocatorV2::AllocationLayoutRegistry::Remove(
    Allocation* allocation, void* ptr) {
  std::lock_guard<SpinLock> guard(spinlock_);
  layouts_by_allocation_.erase(allocation);
  auto ptr_it = layouts_by_ptr_.find(ptr);
  if (ptr_it != layouts_by_ptr_.end() &&
      ptr_it->second.allocation == allocation) {
    layouts_by_ptr_.erase(ptr_it);
  }
}

CUDAVirtualMemAllocatorV2::CUDAVirtualMemAllocatorV2(const GPUPlace& place,
                                                     size_t handle_size,
                                                     PoolType pool)
    : place_(place), handle_size_(handle_size), pool_type_(pool) {}

CUDAVirtualMemAllocatorV2::~CUDAVirtualMemAllocatorV2() {
#if defined(__linux__)
  for (const auto& item : ipc_export_fds_) {
    if (item.second >= 0) {
      ::close(item.second);
    }
  }
#endif
}

bool CUDAVirtualMemAllocatorV2::IsAllocThreadSafe() const { return false; }

void CUDAVirtualMemAllocatorV2::RollbackCreatedHandles(
    const HandleLayout& layout) const {
  for (const auto& meta : layout) {
    if (meta == nullptr) {
      continue;
    }
    phi::dynload::cuMemUnmap(meta->base(), meta->size());
    platform::RecordedGpuMemRelease(
        meta->handle(), meta->size(), place_.device);
  }
}

void CUDAVirtualMemAllocatorV2::MarkLayoutMapped(const HandleLayout& layout) {
  for (const auto& meta : layout) {
    backing_map_.MarkMapped(meta->base(), meta, meta->size());
  }
}

void CUDAVirtualMemAllocatorV2::MarkRemapDestinationLayoutMapped(
    const HandleLayout& layout) {
  for (const auto& meta : layout) {
    // Keep the staged allocation from releasing a transferred handle if the
    // ownership sink throws. Commit clears only this temporary meta marker;
    // the BackingMap destination marker remains until stale-range cleanup.
    meta->MarkOwnedByRemapDestination();
    backing_map_.MarkRemapDestinationMapped(meta->base(), meta, meta->size());
  }
}

void CUDAVirtualMemAllocatorV2::InitOnce() {
  std::call_once(init_flag_, [this] {
    platform::CUDADeviceGuard guard(place_.device);
    prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop_.location.id = place_.device;
#if defined(_WIN32)
    prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
#else
    prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemGetAllocationGranularity(
        &granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // V2 uses a per-pool fixed handle size. Unlike V1, the allocator rounds
    // user input up to the device granularity so upper layers can treat every
    // handle in one HandleLayout as a stable fixed-size building block.
    handle_size_ =
        AlignedSize(std::max(handle_size_, granularity_), granularity_);
    size_t actual_avail = 0;
    size_t actual_total = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemGetInfo(&actual_avail, &actual_total));
    const size_t va_multiplier = GetPoolVAMultiplier(pool_type_);
    PADDLE_ENFORCE_LE(va_multiplier,
                      std::numeric_limits<size_t>::max() / actual_total,
                      common::errors::InvalidArgument(
                          "VA multiplier %d for pool %d overflows size_t.",
                          va_multiplier,
                          static_cast<int>(pool_type_)));
    // Reserve VA by pool to leave room for later split/remap growth. The
    // backing map anchors its handle-sized page grid at the returned base.
    virtual_mem_size_ = AlignedSize(actual_total * va_multiplier, handle_size_);
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemAddressReserve(
        &virtual_mem_base_, virtual_mem_size_, 0, 0, 0));
    backing_map_.Configure(
        virtual_mem_base_, virtual_mem_size_, handle_size_, place_.device);
    CUmemAccessDesc self = {};
    self.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    self.location.id = place_.device;
    self.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc_.push_back(self);
  });
}

phi::Allocation* CUDAVirtualMemAllocatorV2::AllocateImpl(size_t size) {
  return AppendWithLayout(size).allocation.release();
}

CUDAVirtualMemAllocatorV2::AllocationWithLayout
CUDAVirtualMemAllocatorV2::AppendWithLayout(size_t size) {
  InitOnce();
  size_t aligned = AlignedSize(size, handle_size_);
  PADDLE_ENFORCE_LE(
      virtual_mem_alloced_offset_,
      virtual_mem_size_,
      common::errors::InvalidArgument(
          "VMMAllocatorV2 tail offset exceeds reserved VA space."));
  PADDLE_ENFORCE_LE(
      aligned,
      virtual_mem_size_ - virtual_mem_alloced_offset_,
      common::errors::ResourceExhausted("VMMAllocatorV2 virtual address space "
                                        "is exhausted for place %s.",
                                        place_));
  VMMDevicePtr ptr = virtual_mem_base_ + virtual_mem_alloced_offset_;

  auto layout =
      CreateMappedHandleLayout(ptr, aligned, "AppendWithLayout", true);

  MarkLayoutMapped(layout);
  return WrapTrackedAllocation(ptr, aligned, std::move(layout), true);
}

CUDAVirtualMemAllocatorV2::AllocationWithBlock
CUDAVirtualMemAllocatorV2::AppendWithBlock(size_t size) {
  return BuildAllocationWithBlock(AppendWithLayout(size));
}

CUDAVirtualMemAllocatorV2::AllocationWithLayout
CUDAVirtualMemAllocatorV2::PlaceAtVAWithLayout(VMMDevicePtr ptr, size_t size) {
  InitOnce();
  const size_t aligned = AlignedSize(size, handle_size_);
  PADDLE_ENFORCE_EQ(virtual_mem_base_ + virtual_mem_size_ < virtual_mem_base_,
                    false,
                    common::errors::InvalidArgument(
                        "VMMAllocatorV2 reserved VA range overflows."));
  PADDLE_ENFORCE_GE(
      ptr,
      virtual_mem_base_,
      common::errors::InvalidArgument(
          "VMMAllocatorV2 PlaceAtVA ptr is before reserved VA range."));
  PADDLE_ENFORCE_LT(
      ptr,
      virtual_mem_base_ + virtual_mem_size_,
      common::errors::InvalidArgument(
          "VMMAllocatorV2 PlaceAtVA ptr is outside reserved VA range."));
  PADDLE_ENFORCE_EQ(
      (ptr - virtual_mem_base_) % handle_size_,
      0UL,
      common::errors::InvalidArgument(
          "VMMAllocatorV2 PlaceAtVA requires handle-aligned VA, ptr=%p.",
          reinterpret_cast<void*>(ptr)));
  PADDLE_ENFORCE_LE(
      aligned,
      virtual_mem_base_ + virtual_mem_size_ - ptr,
      common::errors::ResourceExhausted(
          "VMMAllocatorV2 PlaceAtVA range exceeds reserved VA space."));

  auto layout =
      CreateMappedHandleLayout(ptr, aligned, "PlaceAtVAWithLayout", false);

  MarkLayoutMapped(layout);
  return WrapTrackedAllocation(ptr, aligned, std::move(layout), false);
}

CUDAVirtualMemAllocatorV2::AllocationWithBlock
CUDAVirtualMemAllocatorV2::PlaceAtVAWithBlock(VMMDevicePtr ptr, size_t size) {
  return BuildAllocationWithBlock(PlaceAtVAWithLayout(ptr, size));
}

HandleLayout CUDAVirtualMemAllocatorV2::CreateMappedHandleLayout(
    VMMDevicePtr ptr, size_t aligned_size, const char* context, bool is_grow) {
  platform::CUDADeviceGuard guard(place_.device);
  const size_t num_handles = aligned_size / handle_size_;
  HandleLayout layout;
  layout.reserve(num_handles);
  for (size_t i = 0; i < num_handles; ++i) {
    VMMAllocHandle handle;
    auto ce = platform::RecordedGpuMemCreate(
        &handle, handle_size_, &prop_, 0, place_.device);
    if (ce != CUDA_SUCCESS) {
      RollbackCreatedHandles(layout);
      if (ce == CUDA_ERROR_OUT_OF_MEMORY) {
        // BadAlloc may be caught by a retry loop, so do not leave a stale CUDA
        // OOM in the runtime error slot.
        (void)platform::GpuGetLastError();
        auto error = common::errors::ResourceExhausted(
            "%s cuMemCreate failed: out of GPU memory at handle %zu/%zu "
            "(handle_size=%zu).",
            context,
            i,
            num_handles,
            handle_size_);
        if (is_grow) {
          throw VMMGrowOOM(
              error.to_string(),
              __FILE__,
              __LINE__,
              VMMGrowOOMInfo{
                  num_handles, i, handle_size_, place_.device, pool_type_});
        }
        PADDLE_THROW_BAD_ALLOC(error);
      }
      PADDLE_ENFORCE_GPU_SUCCESS(ce);
    }

    const VMMDevicePtr dst = ptr + i * handle_size_;
    auto me = phi::dynload::cuMemMap(dst, handle_size_, 0, handle, 0);
    if (me != CUDA_SUCCESS) {
      platform::RecordedGpuMemRelease(handle, handle_size_, place_.device);
      RollbackCreatedHandles(layout);
      PADDLE_THROW(common::errors::External(
          "%s cuMemMap failed at handle %zu/%zu.", context, i, num_handles));
    }
    layout.push_back(std::make_shared<VMMHandleMeta>(
        VMMHandleMeta{dst, handle_size_, handle, place_.device}));
  }
  try {
    SetAccessOrThrow(ptr, aligned_size, num_handles, context);
  } catch (...) {
    RollbackCreatedHandles(layout);
    throw;
  }
  return layout;
}

void CUDAVirtualMemAllocatorV2::SetAccessOrThrow(VMMDevicePtr ptr,
                                                 size_t aligned_size,
                                                 size_t num_handles,
                                                 const char* context) {
  auto access_result =
      SetAccessInChunks(ptr, aligned_size, handle_size_, access_desc_);
  if (access_result.status == CUDA_SUCCESS) {
    return;
  }

  size_t actual_avail = 0;
  size_t actual_total = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemGetInfo(&actual_avail, &actual_total));
  if (access_result.status == CUDA_ERROR_OUT_OF_MEMORY) {
    (void)platform::GpuGetLastError();
    PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
        "%s cuMemSetAccess failed: out of GPU memory at offset %zu/%zu "
        "(failed_size=%zu, handle_size=%zu, handle_count=%zu, "
        "actual_avail=%zu, actual_total=%zu).",
        context,
        access_result.failed_offset,
        aligned_size,
        access_result.failed_size,
        handle_size_,
        num_handles,
        actual_avail,
        actual_total));
  }
  PADDLE_THROW(common::errors::External(
      "%s cuMemSetAccess failed at offset %zu/%zu (failed_size=%zu, "
      "handle_size=%zu, handle_count=%zu, status=%d, actual_avail=%zu, "
      "actual_total=%zu).",
      context,
      access_result.failed_offset,
      aligned_size,
      access_result.failed_size,
      handle_size_,
      num_handles,
      static_cast<int>(access_result.status),
      actual_avail,
      actual_total));
}

bool CUDAVirtualMemAllocatorV2::CollectAllocationHandleLayout(
    void* ptr, HandleLayout* layout) const {
  return allocation_layouts_.Lookup(ptr, layout);
}

bool CUDAVirtualMemAllocatorV2::IsRemapDestinationOwnedLayout(
    const HandleLayout& layout) const {
  return std::all_of(layout.begin(), layout.end(), [](const auto& handle) {
    return handle != nullptr && handle->IsOwnedByRemapDestination();
  });
}

bool CUDAVirtualMemAllocatorV2::IsRemapDestinationAllocation(void* ptr) const {
  HandleLayout layout;
  if (!CollectAllocationHandleLayout(ptr, &layout)) {
    return false;
  }
  return IsRemapDestinationOwnedLayout(layout);
}

bool CUDAVirtualMemAllocatorV2::ClearRemapDestinationOwnership(VMMDevicePtr ptr,
                                                               size_t size) {
  if (!IsReservedVARange(ptr, size)) {
    return false;
  }
  return backing_map_.ClearRemapDestinationOwnership(ptr, size);
}

size_t CUDAVirtualMemAllocatorV2::ClearRemapDestinationOwnershipInRange(
    VMMDevicePtr ptr, size_t size) {
  if (!IsReservedVARange(ptr, size)) {
    return 0;
  }
  return backing_map_.ClearRemapDestinationOwnershipInRange(ptr, size);
}

void CUDAVirtualMemAllocatorV2::FreeImpl(phi::Allocation* allocation) {
  auto* ptr = allocation->ptr();
  HandleLayout layout =
      RequireHandleLayout(static_cast<Allocation*>(allocation));

  int prev_id = -1;
  auto runtime_status = cudaGetDevice(&prev_id);
  if (IsCudaRuntimeDeinitialized(runtime_status)) {
    // CUDA teardown path only. At this point the runtime/context may no longer
    // accept cuMemUnmap/cuMemRelease, and this allocator will not serve further
    // allocations/remap/IPC queries. Drop host-side ownership records to avoid
    // throwing from process-exit cleanup; do not treat this as a normal release
    // path or as evidence that backing_map_ still reflects driver state.
    UnregisterHandleLayout(static_cast<Allocation*>(allocation), ptr);
    delete allocation;
    return;
  }
  PADDLE_ENFORCE_GPU_SUCCESS(runtime_status);
  const bool restore_device = prev_id != place_.device;
  if (restore_device) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(place_.device));
  }
  DEFINE_PADDLE_SCOPE_GUARD([&] {
    if (restore_device) {
      auto status = cudaSetDevice(prev_id);
      if (status != cudaSuccess && !IsCudaRuntimeDeinitialized(status)) {
        VLOG(3) << "Failed to restore CUDA device from " << place_.device
                << " to " << prev_id << ": " << cudaGetErrorString(status);
      }
    }
  });

  for (const auto& handle : layout) {
    if (handle->IsOwnedByRemapDestination()) {
      continue;
    }
    auto unmap_status =
        phi::dynload::cuMemUnmap(handle->base(), handle->size());
    if (IsCudaDeinitialized(unmap_status)) {
      // Terminal cleanup after CUDA teardown. The remaining driver state is no
      // longer observable through CUDA APIs, so backing_map_ is intentionally
      // left untouched; normal runtime failures are still enforced below.
      continue;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(unmap_status);
    auto release_status = platform::RecordedGpuMemRelease(
        handle->handle(), handle->size(), place_.device);
    if (IsCudaDeinitialized(release_status)) {
      // Same terminal-cleanup rule as the unmap path above. Only CUDA teardown
      // errors are tolerated here; other release failures must be surfaced.
      continue;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(release_status);
    CloseIPCExportFD(handle->handle());
    backing_map_.MarkUnmapped(handle->base(), handle->size());
    backing_map_.MarkReleased(handle->base(), handle->handle(), handle->size());
  }

  UnregisterHandleLayout(static_cast<Allocation*>(allocation), ptr);
  delete allocation;
}

void CUDAVirtualMemAllocatorV2::RollbackMappedHandleRange(VMMDevicePtr ptr,
                                                          size_t handle_count) {
  platform::CUDADeviceGuard guard(place_.device);
  for (size_t rollback = 0; rollback < handle_count; ++rollback) {
    auto mapped_dst = ptr + rollback * handle_size_;
    auto unmap_status = phi::dynload::cuMemUnmap(mapped_dst, handle_size_);
    if (unmap_status != CUDA_SUCCESS) {
      VLOG(3) << "Rollback mapped handle range cuMemUnmap failed at "
              << reinterpret_cast<void*>(mapped_dst)
              << " status=" << unmap_status;
    } else {
      backing_map_.MarkUnmapped(mapped_dst, handle_size_);
    }
  }
}

bool CUDAVirtualMemAllocatorV2::UnmapMappedRangeForRemap(
    VMMDevicePtr ptr, size_t handle_count, MoveBackingPageStats* stats) {
  if (handle_count == 0) {
    return true;
  }
  if (ptr % handle_size_ != 0) {
    VLOG(3) << "UnmapMappedRangeForRemap: unaligned ptr="
            << reinterpret_cast<void*>(ptr) << " handle_size=" << handle_size_;
    return false;
  }
  const size_t size = handle_count * handle_size_;
  platform::CUDADeviceGuard guard(place_.device);
  auto op_start = Clock::now();
  auto unmap_status = phi::dynload::cuMemUnmap(ptr, size);
  if (stats != nullptr) {
    stats->unmap_us += ElapsedMicros(op_start, Clock::now());
    stats->unmap_calls += 1;
  }
  if (unmap_status != CUDA_SUCCESS) {
    VLOG(3) << "UnmapMappedRangeForRemap: cuMemUnmap failed at "
            << reinterpret_cast<void*>(ptr) << " size=" << size
            << " status=" << unmap_status;
    return false;
  }
  op_start = Clock::now();
  backing_map_.MarkUnmapped(ptr, size);
  if (stats != nullptr) {
    stats->metadata_us += ElapsedMicros(op_start, Clock::now());
  }
  return true;
}

bool CUDAVirtualMemAllocatorV2::MoveBackingPage(
    const VMMBackingMap::MappedPage& source,
    const VMMBackingMap::UnmappedPage& target,
    MoveBackingPageStats* stats,
    bool defer_target_access,
    bool source_already_unmapped) {
  if ((!source_already_unmapped &&
       !ValidateMappedPages({source}, "MoveBackingPage::source")) ||
      !ValidateUnmappedPages({target}, "MoveBackingPage::target")) {
    return false;
  }
  platform::CUDADeviceGuard guard(place_.device);

  auto op_start = Clock::now();
  if (!source_already_unmapped) {
    auto unmap_source_status =
        phi::dynload::cuMemUnmap(source.va, handle_size_);
    if (stats != nullptr) {
      stats->unmap_us += ElapsedMicros(op_start, Clock::now());
      stats->unmap_calls += 1;
    }
    if (unmap_source_status != CUDA_SUCCESS) {
      VLOG(3) << "MoveBackingPage: source cuMemUnmap failed at "
              << reinterpret_cast<void*>(source.va)
              << " status=" << unmap_source_status;
      return false;
    }
    op_start = Clock::now();
    backing_map_.MarkUnmapped(source.va, handle_size_);
    if (stats != nullptr) {
      stats->metadata_us += ElapsedMicros(op_start, Clock::now());
    }
  }

  auto restore_source = [&]() {
    auto restore_start = Clock::now();
    auto restore_status =
        phi::dynload::cuMemMap(source.va, handle_size_, 0, source.handle, 0);
    if (restore_status != CUDA_SUCCESS) {
      VLOG(3) << "MoveBackingPage: failed to restore source mapping at "
              << reinterpret_cast<void*>(source.va)
              << " status=" << restore_status;
      return false;
    }
    auto access_result =
        SetAccessInChunks(source.va, handle_size_, handle_size_, access_desc_);
    if (access_result.status != CUDA_SUCCESS) {
      VLOG(3) << "MoveBackingPage: failed to restore source access at "
              << reinterpret_cast<void*>(source.va)
              << " failed_offset=" << access_result.failed_offset
              << " failed_size=" << access_result.failed_size
              << " status=" << access_result.status;
      phi::dynload::cuMemUnmap(source.va, handle_size_);
      if (stats != nullptr) {
        stats->restore_us += ElapsedMicros(restore_start, Clock::now());
      }
      return false;
    }
    if (source.meta != nullptr) {
      backing_map_.MarkMapped(source.va, source.meta, handle_size_);
    } else {
      backing_map_.MarkMapped(source.va, source.handle, handle_size_);
    }
    if (stats != nullptr) {
      stats->restore_us += ElapsedMicros(restore_start, Clock::now());
    }
    return true;
  };

  op_start = Clock::now();
  auto map_target_status =
      phi::dynload::cuMemMap(target.va, handle_size_, 0, source.handle, 0);
  if (stats != nullptr) {
    stats->map_us += ElapsedMicros(op_start, Clock::now());
  }
  if (map_target_status != CUDA_SUCCESS) {
    VLOG(3) << "MoveBackingPage: target cuMemMap failed at "
            << reinterpret_cast<void*>(target.va)
            << " status=" << map_target_status;
    if (!source_already_unmapped && !restore_source()) {
      VLOG(3) << "MoveBackingPage: source restore also failed after target "
                 "cuMemMap failure; source VA remains unmapped, source="
              << reinterpret_cast<void*>(source.va)
              << " target=" << reinterpret_cast<void*>(target.va)
              << " handle=" << reinterpret_cast<void*>(source.handle);
    }
    return false;
  }

  if (!defer_target_access) {
    op_start = Clock::now();
    auto access_result =
        SetAccessInChunks(target.va, handle_size_, handle_size_, access_desc_);
    if (stats != nullptr) {
      stats->set_access_us += ElapsedMicros(op_start, Clock::now());
      stats->set_access_calls += access_result.call_count;
    }
    if (access_result.status != CUDA_SUCCESS) {
      VLOG(3) << "MoveBackingPage: target cuMemSetAccess failed at "
              << reinterpret_cast<void*>(target.va)
              << " failed_offset=" << access_result.failed_offset
              << " failed_size=" << access_result.failed_size
              << " status=" << access_result.status;
      op_start = Clock::now();
      auto unmap_target_status =
          phi::dynload::cuMemUnmap(target.va, handle_size_);
      if (stats != nullptr) {
        stats->rollback_us += ElapsedMicros(op_start, Clock::now());
      }
      if (unmap_target_status != CUDA_SUCCESS) {
        VLOG(3) << "MoveBackingPage: target rollback cuMemUnmap failed at "
                << reinterpret_cast<void*>(target.va)
                << " status=" << unmap_target_status;
      }
      if (!source_already_unmapped && !restore_source()) {
        VLOG(3) << "MoveBackingPage: source restore also failed after target "
                   "cuMemSetAccess failure; source VA remains unmapped, source="
                << reinterpret_cast<void*>(source.va)
                << " target=" << reinterpret_cast<void*>(target.va)
                << " handle=" << reinterpret_cast<void*>(source.handle);
      }
      return false;
    }
  }

  op_start = Clock::now();
  if (source.meta != nullptr) {
    backing_map_.MarkMapped(target.va, source.meta, handle_size_);
  } else {
    backing_map_.MarkMapped(target.va, source.handle, handle_size_);
  }
  if (stats != nullptr) {
    stats->metadata_us += ElapsedMicros(op_start, Clock::now());
  }
  return true;
}

bool CUDAVirtualMemAllocatorV2::MoveBackingPageForRemap(
    const VMMBackingMap::MappedPage& source,
    const VMMBackingMap::UnmappedPage& target,
    const std::shared_ptr<VMMHandleMeta>& meta,
    MoveBackingPageStats* stats,
    bool defer_target_access,
    bool source_already_unmapped) {
  if (meta == nullptr) {
    VLOG(3) << "MoveBackingPageForRemap: missing handle metadata for source "
            << reinterpret_cast<void*>(source.va)
            << " target=" << reinterpret_cast<void*>(target.va)
            << " handle=" << reinterpret_cast<void*>(source.handle);
    return false;
  }
  if (!MoveBackingPage(source,
                       target,
                       stats,
                       defer_target_access,
                       source_already_unmapped)) {
    return false;
  }
  auto op_start = Clock::now();
  if (!meta->IsOwnedByRemapDestination()) {
    meta->MarkOwnedByRemapDestination();
  }
  if (stats != nullptr) {
    stats->metadata_us += ElapsedMicros(op_start, Clock::now());
  }
  return true;
}

bool CUDAVirtualMemAllocatorV2::SetAccessForMappedRange(
    VMMDevicePtr ptr, size_t size, MoveBackingPageStats* stats) {
  if (size == 0) {
    return true;
  }
  if (ptr % handle_size_ != 0 || size % handle_size_ != 0) {
    VLOG(3) << "SetAccessForMappedRange: unaligned range ptr="
            << reinterpret_cast<void*>(ptr) << " size=" << size
            << " handle_size=" << handle_size_;
    return false;
  }
  platform::CUDADeviceGuard guard(place_.device);
  auto op_start = Clock::now();
  auto access_result = SetAccessWholeRange(ptr, size, access_desc_);
  if (access_result.status != CUDA_SUCCESS) {
    VLOG(4) << "SetAccessForMappedRange: whole-range cuMemSetAccess failed at "
            << reinterpret_cast<void*>(ptr) << " size=" << size
            << " status=" << access_result.status
            << ", falling back to chunked access";
    access_result = SetAccessInChunks(ptr, size, handle_size_, access_desc_);
  }
  if (stats != nullptr) {
    stats->set_access_us += ElapsedMicros(op_start, Clock::now());
    stats->set_access_calls += access_result.call_count;
  }
  if (access_result.status != CUDA_SUCCESS) {
    VLOG(3) << "SetAccessForMappedRange: cuMemSetAccess failed at "
            << reinterpret_cast<void*>(ptr)
            << " failed_offset=" << access_result.failed_offset
            << " failed_size=" << access_result.failed_size
            << " status=" << access_result.status;
    return false;
  }
  return true;
}

CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult
CUDAVirtualMemAllocatorV2::RestoreRemapSourceMapping(
    VMMAllocHandle handle,
    const std::shared_ptr<VMMHandleMeta>& meta,
    size_t size) {
  if (meta == nullptr || !meta->IsOwnedByRemapDestination()) {
    return RestoreRemapSourceResult::kSkipped;
  }

  platform::CUDADeviceGuard guard(place_.device);
  const VMMDevicePtr original_va = meta->base();
  auto map_status = phi::dynload::cuMemMap(original_va, size, 0, handle, 0);
  if (map_status != CUDA_SUCCESS) {
    VLOG(3) << "RestoreRemapSourceMapping: cuMemMap("
            << reinterpret_cast<void*>(original_va)
            << ") failed status=" << map_status << ", force-releasing handle";
    return ForceReleaseRemapSource(handle, meta, size, "cuMemMap", false);
  }

  auto access_result =
      SetAccessInChunks(original_va, size, handle_size_, access_desc_);
  if (access_result.status != CUDA_SUCCESS) {
    VLOG(3) << "RestoreRemapSourceMapping: cuMemSetAccess failed for VA "
            << reinterpret_cast<void*>(original_va)
            << " failed_offset=" << access_result.failed_offset
            << " failed_size=" << access_result.failed_size
            << " status=" << access_result.status;
    phi::dynload::cuMemUnmap(original_va, size);
    return ForceReleaseRemapSource(handle, meta, size, "cuMemSetAccess", false);
  }

  backing_map_.MarkMapped(original_va, meta, size);
  return RestoreRemapSourceResult::kRestored;
}

CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult
CUDAVirtualMemAllocatorV2::ForceReleaseRemapSource(
    VMMAllocHandle handle,
    const std::shared_ptr<VMMHandleMeta>& meta,
    size_t size,
    const char* context,
    bool unmap_mapped_source) {
  platform::CUDADeviceGuard guard(place_.device);
  const VMMDevicePtr original_va = meta->base();
  if (unmap_mapped_source) {
    auto unmap_status = phi::dynload::cuMemUnmap(original_va, size);
    if (unmap_status == CUDA_SUCCESS) {
      backing_map_.MarkUnmapped(original_va, size);
    } else {
      VLOG(3) << "RestoreRemapSourceMapping: force-release rollback "
              << "cuMemUnmap failed for VA "
              << reinterpret_cast<void*>(original_va)
              << " status=" << unmap_status;
    }
  }

  auto release_status =
      platform::RecordedGpuMemRelease(handle, size, place_.device);
  if (release_status == CUDA_SUCCESS) {
    CloseIPCExportFD(handle);
    backing_map_.MarkReleased(original_va, handle, size);
  } else {
    VLOG(3) << "RestoreRemapSourceMapping: force-release after "
            << (context == nullptr ? "unknown" : context)
            << " failure returned status=" << release_status;
  }
  return RestoreRemapSourceResult::kForceReleased;
}

CUDAVirtualMemAllocatorV2::AllocationWithLayout
CUDAVirtualMemAllocatorV2::WrapTrackedAllocation(VMMDevicePtr ptr,
                                                 size_t size,
                                                 HandleLayout layout,
                                                 bool advance_tail) {
  if (advance_tail) {
    AdvanceTailOffset(size);
  }
  AllocationWithLayout result;
  auto* alloc = CreateTrackedAllocation(ptr, size, layout);
  CUDAVirtualMemAllocatorV2* self = this;
  result.layout = std::move(layout);
  result.allocation = DecoratedAllocationPtr(alloc, [self](phi::Allocation* a) {
    self->FreeImpl(static_cast<Allocation*>(a));
  });
  return result;
}

CUDAVirtualMemAllocatorV2::AllocationWithBlock
CUDAVirtualMemAllocatorV2::BuildAllocationWithBlock(
    AllocationWithLayout allocation_with_layout) {
  AllocationWithBlock result;
  result.block =
      BlockV2::MakeMappedBlock(BlockType::kFree,
                               allocation_with_layout.allocation->ptr(),
                               allocation_with_layout.allocation->size(),
                               pool_type_);
  result.allocation = std::move(allocation_with_layout.allocation);
  return result;
}

Allocation* CUDAVirtualMemAllocatorV2::CreateTrackedAllocation(
    VMMDevicePtr ptr, size_t size, const HandleLayout& layout) {
  auto* allocation = new Allocation(  // NOLINT
      reinterpret_cast<void*>(ptr),
      size,
      place_);
  RegisterHandleLayout(allocation, reinterpret_cast<void*>(ptr), layout);
  return allocation;
}

void CUDAVirtualMemAllocatorV2::RegisterHandleLayout(
    Allocation* allocation, void* ptr, const HandleLayout& layout) {
  allocation_layouts_.Add(allocation, ptr, layout);
  if (!backing_map_.ValidateLayout(layout, "RegisterHandleLayout")) {
    VLOG(3) << "VMM V2 BackingMap validation failed while registering layout "
            << ptr;
  }
}

HandleLayout CUDAVirtualMemAllocatorV2::RequireHandleLayout(
    Allocation* allocation) const {
  HandleLayout layout;
  const bool found = allocation_layouts_.Lookup(allocation, &layout);
  PADDLE_ENFORCE_EQ(
      found,
      true,
      common::errors::NotFound(
          "No VMMAllocatorV2 handle layout found for allocation %p.",
          allocation));
  return layout;
}

void CUDAVirtualMemAllocatorV2::UnregisterHandleLayout(Allocation* allocation,
                                                       void* ptr) {
  allocation_layouts_.Remove(allocation, ptr);
}

CUDAVirtualMemAllocatorV2::StagedRemapDestination
CUDAVirtualMemAllocatorV2::CreateStagedRemapDestination(
    VMMDevicePtr ptr,
    const std::vector<VMMBackingMap::MappedPage>& source_pages,
    size_t start,
    size_t count,
    PoolType pool_type) {
  PADDLE_ENFORCE_LE(start,
                    source_pages.size(),
                    common::errors::InvalidArgument(
                        "Remap source start %zu exceeds page count %zu.",
                        start,
                        source_pages.size()));
  PADDLE_ENFORCE_LE(count,
                    source_pages.size() - start,
                    common::errors::InvalidArgument(
                        "Remap source range [%zu, %zu) exceeds page count %zu.",
                        start,
                        start + count,
                        source_pages.size()));
  HandleLayout layout;
  layout.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    auto meta = std::make_shared<VMMHandleMeta>(
        VMMHandleMeta{ptr + i * handle_size_,
                      handle_size_,
                      source_pages[start + i].handle,
                      place_.device});
    layout.push_back(std::move(meta));
  }

  StagedRemapDestination result;
  const size_t bytes = count * handle_size_;
  try {
    result.allocation = CreateTrackedAllocation(ptr, bytes, layout);
    result.block = BlockV2::MakeMappedBlock(
        BlockType::kFree, reinterpret_cast<void*>(ptr), bytes, pool_type);
    MarkRemapDestinationLayoutMapped(layout);
  } catch (...) {
    DestroyStagedDestinationAllocation(result.allocation);
    result.allocation = nullptr;
    throw;
  }
  return result;
}

DecoratedAllocationPtr
CUDAVirtualMemAllocatorV2::AdoptRemapDestinationAllocation(
    Allocation* allocation) {
  // Use a custom deleter that calls FreeImpl directly, since the
  // The remap destination bypasses the normal Allocate() path and
  // cannot use RegisterDecoratedAllocator (which is private).
  CUDAVirtualMemAllocatorV2* self = this;
  return DecoratedAllocationPtr(allocation, [self](phi::Allocation* a) {
    self->FreeImpl(static_cast<Allocation*>(a));
  });
}

void CUDAVirtualMemAllocatorV2::DestroyStagedDestinationAllocation(
    Allocation* allocation) {
  if (allocation == nullptr) {
    return;
  }
  UnregisterHandleLayout(allocation, allocation->ptr());
  delete allocation;
}

bool CUDAVirtualMemAllocatorV2::HasIPCExportedRange(VMMDevicePtr ptr,
                                                    size_t size) const {
  if (!IsReservedVARange(ptr, size)) {
    return false;
  }
  return backing_map_.HasIPCExportedPages(ptr, size);
}

size_t CUDAVirtualMemAllocatorV2::CountIPCExportedBytes(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const {
  return backing_map_.CountIPCExportedBytes(ranges);
}

bool CUDAVirtualMemAllocatorV2::IsRangeUnmapped(VMMDevicePtr ptr,
                                                size_t size) const {
  return backing_map_.IsRangeUnmapped(ptr, size);
}

bool CUDAVirtualMemAllocatorV2::IsRangeReleasable(VMMDevicePtr ptr,
                                                  size_t size) const {
  return backing_map_.IsRangeReleasable(ptr, size);
}

bool CUDAVirtualMemAllocatorV2::IsReservedVARange(VMMDevicePtr ptr,
                                                  size_t size) const {
  if (size == 0 || ptr < virtual_mem_base_) {
    return false;
  }
  const VMMDevicePtr offset = ptr - virtual_mem_base_;
  return offset <= virtual_mem_size_ && size <= virtual_mem_size_ - offset;
}

bool CUDAVirtualMemAllocatorV2::CollectIPCParts(
    VMMDevicePtr ptr, size_t size, std::vector<BlockPart>* ipc_parts) const {
  std::vector<IPCPartDescriptor> descriptors;
  if (!backing_map_.CollectIPCPartDescriptors(
          ptr, size, ipc_parts != nullptr ? &descriptors : nullptr)) {
    return false;
  }
  if (ipc_parts != nullptr) {
    BuildIPCParts(descriptors, false, ipc_parts);
  }
  return true;
}

bool CUDAVirtualMemAllocatorV2::ExportIPCParts(
    VMMDevicePtr ptr, size_t size, std::vector<BlockPart>* ipc_parts) {
  std::vector<IPCPartDescriptor> descriptors;
  if (!backing_map_.CollectIPCPartDescriptors(
          ptr, size, ipc_parts != nullptr ? &descriptors : nullptr)) {
    return false;
  }
  if (ipc_parts != nullptr) {
    BuildIPCParts(descriptors, true, ipc_parts);
  }
  // Pin the backing only after all fallible FD and metadata construction has
  // completed. A failed export must not make the range permanently
  // unreleasable when no IPC metadata was returned to the caller.
  backing_map_.MarkIPCExported(ptr, size);
  return true;
}

void CUDAVirtualMemAllocatorV2::BuildIPCParts(
    const std::vector<IPCPartDescriptor>& descriptors,
    bool include_shared_fd,
    std::vector<BlockPart>* ipc_parts) const {
  std::vector<BlockPart> collected;
  collected.reserve(descriptors.size());
  for (const auto& descriptor : descriptors) {
    const int shared_fd =
        include_shared_fd ? GetOrCreateIPCExportFD(descriptor.handle) : -1;
    auto chunk = std::make_shared<VmmChunkMeta>(VmmChunkMeta{
        descriptor.handle_base,
        descriptor.handle_size,
        descriptor.handle,
        descriptor.device,
        shared_fd,
    });
    collected.push_back(
        BlockPart{std::move(chunk), descriptor.handle_rel_off, descriptor.len});
  }
  *ipc_parts = std::move(collected);
}

int CUDAVirtualMemAllocatorV2::GetOrCreateIPCExportFD(
    VMMAllocHandle handle) const {
#if defined(__linux__)
  std::lock_guard<SpinLock> guard(ipc_export_lock_);
  auto found = ipc_export_fds_.find(handle);
  if (found != ipc_export_fds_.end()) {
    return found->second;
  }
  int fd = -1;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemExportToShareableHandle(
      &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  try {
    ipc_export_fds_.emplace(handle, fd);
  } catch (...) {
    ::close(fd);
    throw;
  }
  return fd;
#else
  return -1;
#endif
}

void CUDAVirtualMemAllocatorV2::CloseIPCExportFD(VMMAllocHandle handle) const {
#if defined(__linux__)
  int fd = -1;
  {
    std::lock_guard<SpinLock> guard(ipc_export_lock_);
    auto found = ipc_export_fds_.find(handle);
    if (found == ipc_export_fds_.end()) {
      return;
    }
    fd = found->second;
    ipc_export_fds_.erase(found);
  }
  if (fd >= 0) {
    ::close(fd);
  }
#else
  (void)handle;
#endif
}

bool CUDAVirtualMemAllocatorV2::SetBlockRemapEvent(
    const BlockV2& block,
    gpuStream_t stream,
    std::shared_ptr<CUDAEventGuard> event) {
  if (!IsReservedVARange(block.begin_va(), block.size())) {
    return false;
  }
  return SetRemapEvent(block.begin_va(), block.size(), stream, event);
}

bool CUDAVirtualMemAllocatorV2::SetRemapEvent(
    VMMDevicePtr ptr,
    size_t size,
    gpuStream_t stream,
    std::shared_ptr<CUDAEventGuard> event) {
  return backing_map_.MarkPendingEventForRange(
      ptr, size, stream, std::move(event));
}

std::vector<VMMBackingMap::MappedPage>
CUDAVirtualMemAllocatorV2::CollectMappedPages(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  if (target_bytes == 0) {
    return backing_map_.CollectMappedPagesFullyInRange(ranges);
  }
  return backing_map_.CollectMappedPagesFullyInRange(ranges, target_bytes);
}

std::vector<VMMBackingMap::MappedPage>
CUDAVirtualMemAllocatorV2::CollectRemapSourcePages(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  return backing_map_.CollectRemapSourcePagesFullyInRange(ranges, target_bytes);
}

std::vector<VMMBackingMap::UnmappedPage>
CUDAVirtualMemAllocatorV2::CollectUnmappedPages(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  if (target_bytes == 0) {
    return backing_map_.CollectUnmappedPagesFullyInRange(ranges);
  }
  return backing_map_.CollectUnmappedPagesFullyInRange(ranges, target_bytes);
}

bool CUDAVirtualMemAllocatorV2::ValidateMappedPages(
    const std::vector<VMMBackingMap::MappedPage>& pages,
    const char* context) const {
  return backing_map_.ValidateMappedPages(pages, context);
}

bool CUDAVirtualMemAllocatorV2::ValidateUnmappedPages(
    const std::vector<VMMBackingMap::UnmappedPage>& pages,
    const char* context) const {
  return backing_map_.ValidateUnmappedPages(pages, context);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
