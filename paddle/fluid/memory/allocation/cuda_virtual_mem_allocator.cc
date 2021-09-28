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

#include "paddle/fluid/memory/allocation/cuda_virtual_mem_allocator.h"
#include "paddle/fluid/platform/dynload/cuda_driver.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include <string>
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"

#if CUDA_VERSION >= 10020

namespace paddle {
namespace memory {
namespace allocation {

#define PADDLE_ENFORCE_CUDA_SUCCESS2(COND)                               \
  do {                                                                   \
    auto __cond__ = (COND);                                              \
    if (UNLIKELY(__cond__ != CUDA_SUCCESS)) {                            \
      auto __summary__ =                                                 \
          ::paddle::platform::errors::External("cu error %d", __cond__); \
      __THROW_ERROR_INTERNAL__(__summary__);                             \
    }                                                                    \
  } while (0)

CUDAVirtualMemAllocator::CUDAVirtualMemAllocator(
    const platform::CUDAPlace& place)
    : place_(place) {
  prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop_.location.id = place.GetDeviceId();

  access_desc_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc_.location.id = place.GetDeviceId();
  access_desc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  PADDLE_ENFORCE_CUDA_SUCCESS2(
      paddle::platform::dynload::cuMemGetAllocationGranularity(
          &granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  size_t actual_avail, actual_total;
  paddle::platform::CUDADeviceGuard guard(place.GetDeviceId());
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemGetInfo(&actual_avail, &actual_total));

  // virtual_mem_size_ = actual_total/2;

  virtual_mem_size_ = (actual_total + granularity_ - 1) & ~(granularity_ - 1);

  // std::cout << "virtual_mem_size_=" << virtual_mem_size_ << std::endl;

  PADDLE_ENFORCE_CUDA_SUCCESS2(paddle::platform::dynload::cuMemAddressReserve(
      &virtual_mem_base_, virtual_mem_size_, 0, 0, 0));
  virtual_mem_alloced_offset_ = 0;
}

CUDAVirtualMemAllocator::~CUDAVirtualMemAllocator() {
  paddle::platform::CUDADeviceGuard guard(place_.GetDeviceId());
  for (auto& item : virtual_2_physical_map_) {
    PADDLE_ENFORCE_CUDA_SUCCESS2(
        paddle::platform::dynload::cuMemUnmap(item.first, item.second.second));
    PADDLE_ENFORCE_CUDA_SUCCESS2(
        paddle::platform::dynload::cuMemRelease(item.second.first));
  }

  PADDLE_ENFORCE_CUDA_SUCCESS2(paddle::platform::dynload::cuMemAddressFree(
      virtual_mem_base_, virtual_mem_size_));
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

  paddle::platform::CUDADeviceGuard guard(place_.GetDeviceId());
  PADDLE_ENFORCE_CUDA_SUCCESS2(
      paddle::platform::dynload::cuMemUnmap(iter->first, iter->second.second));
  PADDLE_ENFORCE_CUDA_SUCCESS2(
      paddle::platform::dynload::cuMemRelease(iter->second.first));

  virtual_2_physical_map_.erase(iter);

  delete allocation;
}

Allocation* CUDAVirtualMemAllocator::AllocateImpl(size_t size) {
  size = (size + granularity_ - 1) & ~(granularity_ - 1);

  CUdeviceptr ptr = virtual_mem_base_ + virtual_mem_alloced_offset_;

  if (ptr + size > virtual_mem_base_ + virtual_mem_size_) {
    PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
        "\n\nOut of memory error on GPU %d. "
        "Cannot allocate %s memory on GPU %d, %s memory has been allocated and "
        "available memory is only %s.\n\n"
        "Please check whether there is any other process using GPU %d.\n"
        "1. If yes, please stop them, or start PaddlePaddle on another GPU.\n"
        "2. If no, please decrease the batch size of your model.\n\n",
        place_.device, string::HumanReadableSize(size), place_.device,
        string::HumanReadableSize(virtual_mem_alloced_offset_),
        string::HumanReadableSize(virtual_mem_size_ -
                                  virtual_mem_alloced_offset_),
        place_.device));
    return nullptr;
  }

  CUmemGenericAllocationHandle handle;

  paddle::platform::CUDADeviceGuard guard(place_.GetDeviceId());
  auto ret = paddle::platform::dynload::cuMemCreate(&handle, size, &prop_, 0);

  if (ret != CUDA_SUCCESS) {
    if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
      PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
          "\n\nOut of memory error on GPU %d. "
          "Cannot allocate %s memory on GPU %d, %s memory has been allocated "
          "and "
          "available memory is only %s.\n\n"
          "Please check whether there is any other process using GPU %d.\n"
          "1. If yes, please stop them, or start PaddlePaddle on another GPU.\n"
          "2. If no, please decrease the batch size of your model.\n\n",
          place_.device, string::HumanReadableSize(size), place_.device,
          string::HumanReadableSize(virtual_mem_alloced_offset_),
          string::HumanReadableSize(virtual_mem_size_ -
                                    virtual_mem_alloced_offset_),
          place_.device));
    } else {
      PADDLE_ENFORCE_CUDA_SUCCESS2(ret);
    }
    return nullptr;
  }

  ret = paddle::platform::dynload::cuMemMap(ptr, size, 0, handle, 0);

  if (ret != CUDA_SUCCESS) {
    paddle::platform::dynload::cuMemRelease(handle);
    PADDLE_ENFORCE_CUDA_SUCCESS2(ret);
    return nullptr;
  }

  ret = paddle::platform::dynload::cuMemSetAccess(ptr, size, &access_desc_, 1);

  if (ret != CUDA_SUCCESS) {
    paddle::platform::dynload::cuMemUnmap(ptr, size);
    paddle::platform::dynload::cuMemRelease(handle);
    PADDLE_ENFORCE_CUDA_SUCCESS2(ret);
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
