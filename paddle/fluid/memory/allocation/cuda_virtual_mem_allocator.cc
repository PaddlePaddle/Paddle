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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <string>
#include "paddle/fluid/memory/allocation/cuda_virtual_mem_allocator.h"
#include "paddle/fluid/platform/enforce.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/dynload/cuda_driver.h"
#include "paddle/fluid/platform/gpu_info.h"
#endif
#if CUDA_VERSION >= 10020

namespace paddle {
namespace memory {
namespace allocation {

CUDAVirtualMemAllocator::CUDAVirtualMemAllocator(
    const platform::CUDAPlace& place)
    : place_(place) {
  CUmemAllocationProp prop = {};

  // Setup the properties common for all the chunks
  // The allocations will be device pinned memory.
  // This property structure describes the physical location where the memory
  // will be allocated via cuMemCreate allong with additional properties In this
  // case, the allocation will be pinnded device memory local to a given device.
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = place.device;
  prop_ = prop;

  // Prepare the access descriptor array indicating where and how the backings
  // should be visible.
  for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount(); ++dev_id) {
    if (place.device != dev_id) {
      int capable = 0;
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaDeviceCanAccessPeer(&capable, place.device, dev_id));
      if (!capable) {
        VLOG(1) << "device(" << place.device
                << ") can not access peer to device(" << dev_id << ")";
        continue;
      }
    }
    CUmemAccessDesc access_desc = {};
    // Specify which device we are adding mappings for.
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = dev_id;

    // Specify both read and write access.
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc_.push_back(access_desc);
  }

  // Get the minimum granularity needed for all devices
  // (the max of the minimum granularity of each participating device)
  granularity_ = 0;
  for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount(); ++dev_id) {
    size_t granularity;
    prop.location.id = dev_id;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        paddle::platform::dynload::cuMemGetAllocationGranularity(
            &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    granularity_ = std::max(granularity, granularity_);
  }

  size_t actual_avail, actual_total;
  paddle::platform::CUDADeviceGuard guard(place.device);
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemGetInfo(&actual_avail, &actual_total));

  virtual_mem_size_ = AlignedSize(actual_total, granularity_);

  // Reserve the required contiguous virtual address space for the allocations
  // The maximum video memory size we can apply for is the video memory size of
  // GPU,
  // so the virtual address space size we reserve is equal to the GPU video
  // memory size
  PADDLE_ENFORCE_CUDA_SUCCESS(paddle::platform::dynload::cuMemAddressReserve(
      &virtual_mem_base_, virtual_mem_size_, 0, 0, 0));

  virtual_mem_alloced_offset_ = 0;
}

bool CUDAVirtualMemAllocator::IsAllocThreadSafe() const { return false; }

void CUDAVirtualMemAllocator::FreeImpl(Allocation* allocation) {
  PADDLE_ENFORCE_EQ(
      BOOST_GET_CONST(platform::CUDAPlace, allocation->place()), place_,
      platform::errors::PermissionDenied(
          "GPU memory is freed in incorrect device. This may be a bug"));

  auto iter = virtual_2_physical_map_.find(
      reinterpret_cast<CUdeviceptr>(allocation->ptr()));
  if (iter == virtual_2_physical_map_.end()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Can not find virtual memory address at %s", allocation->ptr()));
  }

  int prev_id;
  cudaGetDevice(&prev_id);
  if (prev_id != place_.device) {
    cudaSetDevice(place_.device);
  }

  auto result =
      paddle::platform::dynload::cuMemUnmap(iter->first, iter->second.second);
  if (result != CUDA_ERROR_DEINITIALIZED) {
    PADDLE_ENFORCE_CUDA_SUCCESS(result);
  }

  if (result != CUDA_ERROR_DEINITIALIZED) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::RecordedCuMemRelease(
        iter->second.first, iter->second.second, place_.device));
  }

  if (prev_id != place_.device) {
    cudaSetDevice(prev_id);
  }

  virtual_2_physical_map_.erase(iter);

  delete allocation;
}

Allocation* CUDAVirtualMemAllocator::AllocateImpl(size_t size) {
  size = AlignedSize(size, granularity_);

  CUdeviceptr ptr = virtual_mem_base_ + virtual_mem_alloced_offset_;

  if (ptr + size > virtual_mem_base_ + virtual_mem_size_) {
    PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
        "\n\nOut of memory error on GPU Virtual Memory %d. "
        "Cannot allocate %s memory on GPU Virtual Memory %d, %s memory has "
        "been allocated and "
        "available memory is only %s.\n\n"
        "Please decrease the batch size of your model.\n\n",
        place_.device, string::HumanReadableSize(size), place_.device,
        string::HumanReadableSize(virtual_mem_alloced_offset_),
        string::HumanReadableSize(virtual_mem_size_ -
                                  virtual_mem_alloced_offset_),
        place_.device));
    return nullptr;
  }

  CUmemGenericAllocationHandle handle;

  paddle::platform::CUDADeviceGuard guard(place_.device);

  // Create physical memory backing allocation.
  auto result =
      platform::RecordedCuMemCreate(&handle, size, &prop_, 0, place_.device);

  if (result != CUDA_SUCCESS) {
    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
      size_t actual_avail, actual_total;
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemGetInfo(&actual_avail, &actual_total));
      size_t actual_allocated = actual_total - actual_avail;

      PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
          "\n\nOut of memory error on GPU %d. "
          "Cannot allocate %s memory on GPU %d, %s memory has been allocated "
          "and "
          "available memory is only %s.\n\n"
          "Please check whether there is any other process using GPU %d.\n"
          "1. If yes, please stop them, or start PaddlePaddle on another GPU.\n"
          "2. If no, please decrease the batch size of your model.\n\n",
          place_.device, string::HumanReadableSize(size), place_.device,
          string::HumanReadableSize(actual_allocated),
          string::HumanReadableSize(actual_avail), place_.device));
    } else {
      PADDLE_ENFORCE_CUDA_SUCCESS(result);
    }
    return nullptr;
  }

  // Assign the chunk to the appropriate VA range and release the handle.
  // After mapping the memory, it can be referenced by virtual address.
  // The allocation will be kept live until it is unmapped.
  result = paddle::platform::dynload::cuMemMap(ptr, size, 0, handle, 0);

  if (result != CUDA_SUCCESS) {
    platform::RecordedCuMemRelease(handle, size, place_.device);
    PADDLE_ENFORCE_CUDA_SUCCESS(result);
    return nullptr;
  }

  // Apply the access descriptors to the whole VA range.
  result = paddle::platform::dynload::cuMemSetAccess(
      ptr, size, access_desc_.data(), access_desc_.size());

  if (result != CUDA_SUCCESS) {
    paddle::platform::dynload::cuMemUnmap(ptr, size);
    platform::RecordedCuMemRelease(handle, size, place_.device);
    PADDLE_ENFORCE_CUDA_SUCCESS(result);
    return nullptr;
  }

  virtual_2_physical_map_.emplace(ptr, std::make_pair(handle, size));

  virtual_mem_alloced_offset_ += size;

  return new Allocation(reinterpret_cast<void*>(ptr), size,
                        platform::Place(place_));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
