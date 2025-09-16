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
#include <unistd.h>  // close
#endif

#include <string>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cuda_driver.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif
#if CUDA_VERSION >= 10020

namespace paddle::memory::allocation {

CUDAVirtualMemAllocator::CUDAVirtualMemAllocator(const phi::GPUPlace& place)
    : place_(place), virtual_mem_base_(0), prop_{} {
  CUmemAllocationProp prop = {};

  // Setup the properties common for all the chunks
  // The allocations will be device pinned memory.
  // This property structure describes the physical location where the memory
  // will be allocated via cuMemCreate along with additional properties In this
  // case, the allocation will be pinned device memory local to a given device.
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = place.device;  // NOLINT
  // Linux：POSIX FD
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop_ = prop;

  // Prepare the access descriptor array indicating where and how the backings
  // should be visible.
  for (int dev_id = 0; dev_id < platform::GetGPUDeviceCount(); ++dev_id) {
    if (place.device != dev_id) {
      int capable = 0;
      PADDLE_ENFORCE_GPU_SUCCESS(
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
  for (int dev_id = 0; dev_id < platform::GetGPUDeviceCount(); ++dev_id) {
    size_t granularity;
    prop.location.id = dev_id;
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    granularity_ = std::max(granularity, granularity_);
  }

  size_t actual_avail, actual_total;
  paddle::platform::CUDADeviceGuard guard(place.device);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemGetInfo(&actual_avail, &actual_total));

  virtual_mem_size_ = AlignedSize(actual_total, granularity_);

  // Reserve the required contiguous virtual address space for the allocations
  // The maximum video memory size we can apply for is the video memory size of
  // GPU,
  // so the virtual address space size we reserve is equal to the GPU video
  // memory size
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuMemAddressReserve(
      &virtual_mem_base_, virtual_mem_size_, 0, 0, 0));

  virtual_mem_alloced_offset_ = 0;
  // ★ 注册到全局：用于后续通过 device 找到 allocator
  // —— 注册到每设备表
  Register(place.device, this);
}

CUDAVirtualMemAllocator::~CUDAVirtualMemAllocator() {
  // 1) 从注册表摘掉自己（防止后续 TryExportShareHandle 命中已销毁实例）
  Unregister(place_.device, this);

  // 2) 可选：在调试期确保没有未解除映射的区域
  if (!virtual_2_physical_map_.empty()) {
    VLOG(2) << "CUDAVirtualMemAllocator destroyed with "
            << virtual_2_physical_map_.size()
            << " mapped regions still tracked.";
  }

  // 3) 可选：释放保留的 VA 空间（如果设计允许在析构时做）
#if CUDA_VERSION >= 10020
  if (virtual_mem_base_ && virtual_mem_size_) {
    auto r =
        phi::dynload::cuMemAddressFree(virtual_mem_base_, virtual_mem_size_);
    VLOG(6) << "cuMemAddressFree(" << reinterpret_cast<void*>(virtual_mem_base_)
            << ", " << virtual_mem_size_ << ") -> " << r;
    (void)r;
    virtual_mem_base_ = 0;
    virtual_mem_size_ = 0;
  }
#endif
}

bool CUDAVirtualMemAllocator::IsAllocThreadSafe() const { return false; }

void CUDAVirtualMemAllocator::FreeImpl(phi::Allocation* allocation) {
  PADDLE_ENFORCE_EQ(
      allocation->place(),
      place_,
      common::errors::PermissionDenied(
          "GPU memory is freed in incorrect device. This may be a bug"));

  auto iter = virtual_2_physical_map_.find(
      reinterpret_cast<CUdeviceptr>(allocation->ptr()));
  if (iter == virtual_2_physical_map_.end()) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Can not find virtual memory address at %s", allocation->ptr()));
  }

  int prev_id;
  cudaGetDevice(&prev_id);
  if (prev_id != place_.device) {
    cudaSetDevice(place_.device);
  }

  auto result = phi::dynload::cuMemUnmap(iter->first, iter->second.second);
  if (result != CUDA_ERROR_DEINITIALIZED) {
    PADDLE_ENFORCE_GPU_SUCCESS(result);
  }

  if (result != CUDA_ERROR_DEINITIALIZED) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::RecordedGpuMemRelease(
        iter->second.first, iter->second.second, place_.device));
  }

  if (prev_id != place_.device) {
    cudaSetDevice(prev_id);
  }

  virtual_2_physical_map_.erase(iter);

  delete allocation;
}

phi::Allocation* CUDAVirtualMemAllocator::AllocateImpl(size_t size) {
  std::call_once(once_flag_, [this] { platform::SetDeviceId(place_.device); });
  size = AlignedSize(size, granularity_);

  CUdeviceptr ptr = virtual_mem_base_ + virtual_mem_alloced_offset_;

  if (ptr + size > virtual_mem_base_ + virtual_mem_size_) {
    PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
        "\n\nOut of memory error on GPU Virtual Memory %d. "
        "Cannot allocate %s memory on GPU Virtual Memory %d, %s memory has "
        "been allocated and "
        "available memory is only %s.\n\n"
        "Please decrease the batch size of your model.\n\n",
        place_.device,
        string::HumanReadableSize(size),
        place_.device,
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
      platform::RecordedGpuMemCreate(&handle, size, &prop_, 0, place_.device);

  if (result != CUDA_SUCCESS) {
    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
      size_t actual_avail, actual_total;
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemGetInfo(&actual_avail, &actual_total));
      size_t actual_allocated = actual_total - actual_avail;

      PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
          "\n\nOut of memory error on GPU %d. "
          "Cannot allocate %s memory on GPU %d, %s memory has been allocated "
          "and "
          "available memory is only %s.\n\n"
          "Please check whether there is any other process using GPU %d.\n"
          "1. If yes, please stop them, or start PaddlePaddle on another GPU.\n"
          "2. If no, please decrease the batch size of your model.\n\n",
          place_.device,
          string::HumanReadableSize(size),
          place_.device,
          string::HumanReadableSize(actual_allocated),
          string::HumanReadableSize(actual_avail),
          place_.device));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(result);
    }
    return nullptr;
  }

  // Assign the chunk to the appropriate VA range and release the handle.
  // After mapping the memory, it can be referenced by virtual address.
  // The allocation will be kept live until it is unmapped.
  result = phi::dynload::cuMemMap(ptr, size, 0, handle, 0);

  if (result != CUDA_SUCCESS) {
    platform::RecordedGpuMemRelease(handle, size, place_.device);
    PADDLE_ENFORCE_GPU_SUCCESS(result);
    return nullptr;
  }

  // Apply the access descriptors to the whole VA range.
  result = phi::dynload::cuMemSetAccess(
      ptr, size, access_desc_.data(), access_desc_.size());

  if (result != CUDA_SUCCESS) {
    phi::dynload::cuMemUnmap(ptr, size);
    platform::RecordedGpuMemRelease(handle, size, place_.device);
    PADDLE_ENFORCE_GPU_SUCCESS(result);
    return nullptr;
  }

  virtual_2_physical_map_.emplace(ptr, std::make_pair(handle, size));

  virtual_mem_alloced_offset_ += size;

  return new Allocation(
      reinterpret_cast<void*>(ptr), size, phi::Place(place_));  // NOLINT
}

// --------- VMM IPC: export/import helpers ----------

// 区间命中 + 导出 FD
bool CUDAVirtualMemAllocator::ExportShareHandleFromVA(CUdeviceptr va,
                                                      VmmShareInfo* out) {
  if (virtual_2_physical_map_.empty()) return false;
  auto it = virtual_2_physical_map_.upper_bound(va);  // first key > va
  if (it == virtual_2_physical_map_.begin()) return false;
  --it;  // now it->first <= va
  CUdeviceptr region_base = it->first;
  size_t region_size = it->second.second;
  if (va < region_base || va >= region_base + region_size) return false;

  int fd = -1;
  auto r = phi::dynload::cuMemExportToShareableHandle(
      &fd, it->second.first, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (r != CUDA_SUCCESS) {
    VLOG(10) << "cuMemExportToShareableHandle failed r=" << r
             << " base=" << reinterpret_cast<void*>(region_base)
             << " size=" << region_size;
    return false;
  }
  out->device = place_.device;
  out->os_fd = fd;
  out->offset = static_cast<size_t>(va - region_base);
  out->size = region_size;
  return true;
}

void CUDAVirtualMemAllocator::Register(int device, CUDAVirtualMemAllocator* a) {
  std::lock_guard<std::mutex> g(s_reg_mu_);
  s_regs_[device].push_back(a);
}

void CUDAVirtualMemAllocator::Unregister(int device,
                                         CUDAVirtualMemAllocator* a) {
  std::lock_guard<std::mutex> g(s_reg_mu_);
  auto it = s_regs_.find(device);
  if (it == s_regs_.end()) return;
  auto& v = it->second;
  v.erase(std::remove(v.begin(), v.end(), a), v.end());
  if (v.empty()) s_regs_.erase(it);
}

bool CUDAVirtualMemAllocator::TryExportShareHandle(int device,
                                                   void* any_va,
                                                   VmmShareInfo* out) {
  std::lock_guard<std::mutex> g(s_reg_mu_);
  auto it = s_regs_.find(device);
  if (it == s_regs_.end()) return false;
  CUdeviceptr va = reinterpret_cast<CUdeviceptr>(any_va);
  for (auto* a : it->second) {
    if (a && a->ExportShareHandleFromVA(va, out)) return true;
  }
  return false;
}

class CudaVmmImportedAllocation final : public Allocation {
 public:
  CudaVmmImportedAllocation(void* ptr,
                            void* base_ptr,
                            size_t size,
                            phi::Place place,
                            CUmemGenericAllocationHandle h,
                            CUdeviceptr va,
                            size_t va_size,
                            int os_fd)
      : Allocation(ptr, base_ptr, size, place),
        handle_(h),
        va_(va),
        va_size_(va_size),
        os_fd_(os_fd) {}

  ~CudaVmmImportedAllocation() override {
#if CUDA_VERSION >= 10020
    auto unmap = phi::dynload::cuMemUnmap(va_, va_size_);
    if (unmap != CUDA_ERROR_DEINITIALIZED) PADDLE_ENFORCE_GPU_SUCCESS(unmap);
    auto rel = phi::dynload::cuMemRelease(handle_);
    if (rel != CUDA_ERROR_DEINITIALIZED) PADDLE_ENFORCE_GPU_SUCCESS(rel);
    phi::dynload::cuMemAddressFree(va_, va_size_);
    if (os_fd_ >= 0) ::close(os_fd_);
#endif
  }

 private:
  CUmemGenericAllocationHandle handle_{};
  CUdeviceptr va_{0};
  size_t va_size_{0};
  int os_fd_{-1};
};

}  // namespace paddle::memory::allocation

#endif
