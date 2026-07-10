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
#include <limits>
#include <utility>

#include "glog/logging.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/scope_guard.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

constexpr size_t kVMMSetAccessChunkSize = 64UL << 20;

bool IsCudaDeinitialized(CUresult result) {
  return result == CUDA_ERROR_DEINITIALIZED;
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
};

SetAccessResult SetAccessInChunks(VMMDevicePtr ptr,
                                  size_t size,
                                  size_t handle_size,
                                  const std::vector<CUmemAccessDesc>& desc) {
  const size_t chunk_size =
      std::max(handle_size, AlignedSize(kVMMSetAccessChunkSize, handle_size));
  size_t offset = 0;
  while (offset < size) {
    const size_t remaining = size - offset;
    const size_t current_size = std::min(chunk_size, remaining);
    auto status = phi::dynload::cuMemSetAccess(
        ptr + offset, current_size, desc.data(), desc.size());
    if (status != CUDA_SUCCESS) {
      return {status, offset, current_size};
    }
    offset += current_size;
  }
  return {};
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
    void* ptr, const HandleLayout& layout) {
  std::lock_guard<SpinLock> guard(spinlock_);
  EmplaceOrEnforce(&layouts_, ptr, layout, "allocation_layout_map_");
}

bool CUDAVirtualMemAllocatorV2::AllocationLayoutRegistry::Lookup(
    void* ptr, HandleLayout* layout) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto it = layouts_.find(ptr);
  if (it == layouts_.end()) {
    return false;
  }
  if (layout != nullptr) {
    *layout = it->second;
  }
  return true;
}

void CUDAVirtualMemAllocatorV2::AllocationLayoutRegistry::Remove(void* ptr) {
  std::lock_guard<SpinLock> guard(spinlock_);
  layouts_.erase(ptr);
}

CUDAVirtualMemAllocatorV2::CUDAVirtualMemAllocatorV2(const GPUPlace& place,
                                                     size_t handle_size,
                                                     PoolType pool)
    : place_(place), handle_size_(handle_size), pool_type_(pool) {}

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
    // Reserve VA by pool to leave room for later split and in-place reuse.
    virtual_mem_size_ = AlignedSize(actual_total * va_multiplier, granularity_);
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

  auto layout = CreateMappedHandleLayout(ptr, aligned, "AppendWithLayout");

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
  const size_t num_handles = aligned / handle_size_;
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

  VLOG(6) << "VMM V2 PlaceAtVA(AllocateAtVA) ptr="
          << reinterpret_cast<void*>(ptr) << " requested=" << size
          << " aligned=" << aligned << " handle_count=" << num_handles
          << " tail_offset=" << virtual_mem_alloced_offset_;
  auto layout = CreateMappedHandleLayout(ptr, aligned, "PlaceAtVAWithLayout");

  MarkLayoutMapped(layout);
  return WrapTrackedAllocation(ptr, aligned, std::move(layout), false);
}

CUDAVirtualMemAllocatorV2::AllocationWithBlock
CUDAVirtualMemAllocatorV2::PlaceAtVAWithBlock(VMMDevicePtr ptr, size_t size) {
  return BuildAllocationWithBlock(PlaceAtVAWithLayout(ptr, size));
}

HandleLayout CUDAVirtualMemAllocatorV2::CreateMappedHandleLayout(
    VMMDevicePtr ptr, size_t aligned_size, const char* context) {
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
        PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
            "%s cuMemCreate failed: out of GPU memory at handle %zu/%zu "
            "(handle_size=%zu).",
            context,
            i,
            num_handles,
            handle_size_));
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

void CUDAVirtualMemAllocatorV2::FreeImpl(phi::Allocation* allocation) {
  auto* ptr = allocation->ptr();
  HandleLayout layout = RequireHandleLayout(ptr);

  int prev_id = -1;
  bool restore_device = false;
  if (cudaGetDevice(&prev_id) == cudaSuccess && prev_id != place_.device) {
    restore_device = cudaSetDevice(place_.device) == cudaSuccess;
  }
  DEFINE_PADDLE_SCOPE_GUARD([&] {
    if (restore_device) {
      cudaSetDevice(prev_id);
    }
  });

  for (const auto& handle : layout) {
    auto result = phi::dynload::cuMemUnmap(handle->base(), handle->size());
    if (IsCudaDeinitialized(result)) {
      continue;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(result);
    backing_map_.MarkUnmapped(handle->base(), handle->size());
    result = platform::RecordedGpuMemRelease(
        handle->handle(), handle->size(), place_.device);
    if (IsCudaDeinitialized(result)) {
      continue;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(result);
    backing_map_.MarkReleased(handle->base(), handle->handle(), handle->size());
  }

  UnregisterHandleLayout(ptr);
  delete allocation;
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
  RegisterHandleLayout(reinterpret_cast<void*>(ptr), layout);
  return new Allocation(reinterpret_cast<void*>(ptr), size, place_);  // NOLINT
}

void CUDAVirtualMemAllocatorV2::RegisterHandleLayout(
    void* ptr, const HandleLayout& layout) {
  allocation_layouts_.Add(ptr, layout);
  if (!backing_map_.ValidateLayout(layout, "RegisterHandleLayout")) {
    VLOG(0) << "VMM V2 BackingMap validation failed while registering layout "
            << ptr;
  }
}

HandleLayout CUDAVirtualMemAllocatorV2::RequireHandleLayout(void* ptr) const {
  HandleLayout layout;
  const bool found = allocation_layouts_.Lookup(ptr, &layout);
  PADDLE_ENFORCE_EQ(
      found,
      true,
      common::errors::NotFound(
          "No VMMAllocatorV2 handle layout found for allocation %p.", ptr));
  return layout;
}

void CUDAVirtualMemAllocatorV2::UnregisterHandleLayout(void* ptr) {
  allocation_layouts_.Remove(ptr);
}

bool CUDAVirtualMemAllocatorV2::IsRangeReleasable(VMMDevicePtr ptr,
                                                  size_t size) const {
  return backing_map_.IsRangeReleasable(ptr, size);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
