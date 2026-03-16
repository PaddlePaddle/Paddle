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

#include "paddle/phi/core/platform/cuda_device_guard.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

size_t GetPoolVAMultiplier(PoolType pool_type) {
  switch (pool_type) {
    case PoolType::kStable:
      return 2;
    case PoolType::kLongLived:
      return 3;
    case PoolType::kTransient:
      return 4;
    case PoolType::kOversized:
      return 1;
  }
  return 1;
}

}  // namespace

CUDAVirtualMemAllocatorV2::CUDAVirtualMemAllocatorV2(const GPUPlace& place,
                                                     size_t handle_size,
                                                     PoolType pool)
    : place_(place), handle_size_(handle_size), pool_type_(pool) {}

bool CUDAVirtualMemAllocatorV2::IsAllocThreadSafe() const { return false; }

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
    // Reserves VA by pool to leave room for later split/remap growth.
    virtual_mem_size_ = AlignedSize(actual_total * va_multiplier, granularity_);
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemAddressReserve(
        &virtual_mem_base_, virtual_mem_size_, 0, 0, 0));
    CUmemAccessDesc self = {};
    self.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    self.location.id = place_.device;
    self.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc_.push_back(self);
  });
}

phi::Allocation* CUDAVirtualMemAllocatorV2::AllocateImpl(size_t size) {
  InitOnce();
  size_t aligned = AlignedSize(size, handle_size_);
  size_t num_handles = aligned / handle_size_;
  VmmDevicePtr ptr = virtual_mem_base_ + virtual_mem_alloced_offset_;
  PADDLE_ENFORCE_LE(
      ptr + aligned,
      virtual_mem_base_ + virtual_mem_size_,
      common::errors::ResourceExhausted("VMMAllocatorV2 virtual address space "
                                        "is exhausted for place %s.",
                                        place_));

  platform::CUDADeviceGuard guard(place_.device);
  HandleLayout layout;
  layout.reserve(num_handles);
  for (size_t i = 0; i < num_handles; ++i) {
    VmmAllocHandle handle;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cuMemCreate(&handle, handle_size_, &prop_, 0));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemMap(
        ptr + i * handle_size_, handle_size_, 0, handle, 0));
    layout.push_back(std::make_shared<VmmHandleMeta>(VmmHandleMeta{
        ptr + i * handle_size_, handle_size_, handle, place_.device}));
  }
  // TODO(zhangting35): Roll back already-created / already-mapped handles if
  // cuMemCreate or cuMemMap fails part way through the loop.
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemSetAccess(
      ptr, aligned, access_desc_.data(), access_desc_.size()));

  RegisterHandleLayout(reinterpret_cast<void*>(ptr), layout);
  AdvanceTailOffset(aligned);
  return new Allocation(reinterpret_cast<void*>(ptr),
                        reinterpret_cast<void*>(ptr),
                        aligned,
                        place_);
}

void CUDAVirtualMemAllocatorV2::FreeImpl(phi::Allocation* allocation) {
  auto* base_ptr = static_cast<Allocation*>(allocation)->base_ptr();
  HandleLayout layout;
  {
    std::lock_guard<std::mutex> guard(base_ptr_layout_mu_);
    auto it = base_ptr_layout_map_.find(base_ptr);
    PADDLE_ENFORCE_NE(
        it == base_ptr_layout_map_.end(),
        true,
        common::errors::NotFound(
            "No VMMAllocatorV2 handle layout found for allocation %p.",
            base_ptr));
    layout = it->second;
  }

  platform::CUDADeviceGuard guard(place_.device);
  for (const auto& handle : layout) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cuMemUnmap(handle->base, handle->size));
    // TODO(zhangting35): Move handle release into shared handle lifetime
    // management once remap / IPC starts sharing one handle across multiple
    // BlockPartV2 objects.
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemRelease(handle->handle));
  }
  UnregisterHandleLayout(base_ptr);
  delete allocation;
}

void CUDAVirtualMemAllocatorV2::UnmapHandle(VmmDevicePtr ptr, size_t size) {
  platform::CUDADeviceGuard guard(place_.device);
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemUnmap(ptr, size));
}

void CUDAVirtualMemAllocatorV2::MapHandlesToVA(
    VmmDevicePtr ptr, const std::vector<VmmAllocHandle>& hs) {
  platform::CUDADeviceGuard guard(place_.device);
  // V2 currently assumes one uniform handle size per pool, so remap can
  // re-materialize a contiguous VA range by replaying fixed-size mappings.
  for (size_t i = 0; i < hs.size(); ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemMap(
        ptr + i * handle_size_, handle_size_, 0, hs[i], 0));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemSetAccess(
      ptr, hs.size() * handle_size_, access_desc_.data(), access_desc_.size()));
}

bool CUDAVirtualMemAllocatorV2::CollectAllocationHandleLayout(
    void* base_ptr, HandleLayout* layout) const {
  std::lock_guard<std::mutex> guard(base_ptr_layout_mu_);
  auto it = base_ptr_layout_map_.find(base_ptr);
  if (it == base_ptr_layout_map_.end()) {
    return false;
  }
  if (layout) {
    *layout = it->second;
  }
  return true;
}

void CUDAVirtualMemAllocatorV2::RegisterHandleLayout(
    void* base_ptr, const HandleLayout& layout) {
  std::lock_guard<std::mutex> guard(base_ptr_layout_mu_);
  base_ptr_layout_map_[base_ptr] = layout;
}

void CUDAVirtualMemAllocatorV2::UnregisterHandleLayout(void* base_ptr) {
  std::lock_guard<std::mutex> guard(base_ptr_layout_mu_);
  base_ptr_layout_map_.erase(base_ptr);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
