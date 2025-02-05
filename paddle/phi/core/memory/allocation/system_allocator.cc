/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include "paddle/phi/core/memory/allocation/system_allocator.h"

#include "paddle/phi/core/memory/stats.h"

#ifdef _WIN32
#include <malloc.h>
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>  // VirtualLock/VirtualUnlock
#else
#include <sys/mman.h>  // for mlock and munlock
#endif

#include "paddle/common/flags.h"
#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/platform/cuda_device_guard.h"
#endif

#include "paddle/phi/core/platform/device/device_wrapper.h"
#include "paddle/phi/core/platform/profiler/mem_tracing.h"

COMMON_DECLARE_bool(use_pinned_memory);
COMMON_DECLARE_double(fraction_of_gpu_memory_to_use);
COMMON_DECLARE_uint64(initial_gpu_memory_in_mb);
COMMON_DECLARE_uint64(reallocate_gpu_memory_in_mb);

namespace paddle::memory::detail {

void* AlignedMalloc(size_t size) {
  void* p = nullptr;
  size_t alignment = 32ul;
#ifdef PADDLE_WITH_DNNL
  // refer to https://github.com/01org/mkl-dnn/blob/master/include/dnnl.hpp
  // memory alignment
  alignment = 4096ul;
#endif
#ifdef _WIN32
  p = _aligned_malloc(size, alignment);
#else
  int error = posix_memalign(&p, alignment, size);
  PADDLE_ENFORCE_EQ(
      error,
      0,
      common::errors::ResourceExhausted(
          "Fail to alloc memory of %ld size, error code is %d.", size, error));
#endif
  PADDLE_ENFORCE_NOT_NULL(p,
                          common::errors::ResourceExhausted(
                              "Fail to alloc memory of %ld size.", size));
  return p;
}

void* CPUAllocator::Alloc(size_t* index, size_t size) {
  // According to http://www.cplusplus.com/reference/cstdlib/malloc/,
  // malloc might not return nullptr if size is zero, but the returned
  // pointer shall not be dereferenced -- so we make it nullptr.
  if (size <= 0) return nullptr;

  *index = 0;  // unlock memory

  void* p = AlignedMalloc(size);

  if (p != nullptr) {
    if (FLAGS_use_pinned_memory) {
      *index = 1;
#ifdef _WIN32
      VirtualLock(p, size);
#else
      mlock(p, size);  // lock memory
#endif
    }
  }

  HOST_MEMORY_STAT_UPDATE(Reserved, 0, size);
  platform::RecordMemEvent(
      p, CPUPlace(), size, phi::TracerMemEventType::ReservedAllocate);
  return p;
}

void CPUAllocator::Free(void* p, size_t size, size_t index) {
  if (p != nullptr && index == 1) {
#ifdef _WIN32
    VirtualUnlock(p, size);
#else
    munlock(p, size);
#endif
  }
  HOST_MEMORY_STAT_UPDATE(Reserved, 0, -size);
  platform::RecordMemEvent(
      p, CPUPlace(), size, phi::TracerMemEventType::ReservedFree);
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);  // NOLINT
#endif
}

bool CPUAllocator::UseGpu() const { return false; }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

void* GPUAllocator::Alloc(size_t* index, size_t size) {
  // CUDA documentation doesn't explain if cudaMalloc returns nullptr
  // if size is 0.  We just make sure it does.
  if (size <= 0) return nullptr;

  void* p;
  auto result = platform::RecordedGpuMalloc(&p, size, gpu_id_);

  if (result == gpuSuccess) {
    *index = 0;
    gpu_alloc_size_ += size;
    return p;
  } else {
    size_t avail, total, actual_avail, actual_total;
    bool is_limited = platform::RecordedGpuMemGetInfo(
        &avail, &total, &actual_avail, &actual_total, gpu_id_);
    size_t allocated = total - avail;

    std::string err_msg;
    if (is_limited) {
      auto limit_size = (total >> 20);
      err_msg = string::Sprintf(
          "\n   3) Set environment variable `FLAGS_gpu_memory_limit_mb` to a "
          "larger value. Currently `FLAGS_gpu_memory_limit_mb` is %d, so the "
          "maximum GPU memory usage is limited to %d MB.\n"
          "      The command is `export FLAGS_gpu_memory_limit_mb=xxx`.",
          limit_size,
          limit_size);
    }

    PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
        "\n\nOut of memory error on GPU %d. "
        "Cannot allocate %s memory on GPU %d, %s memory has been allocated and "
        "available memory is only %s.\n\n"
        "Please check whether there is any other process using GPU %d.\n"
        "1. If yes, please stop them, or start PaddlePaddle on another GPU.\n"
        "2. If no, please try one of the following suggestions:\n"
        "   1) Decrease the batch size of your model.\n"
        "   2) FLAGS_fraction_of_gpu_memory_to_use is %.2lf now, "
        "please set it to a higher value but less than 1.0.\n"
        "      The command is "
        "`export FLAGS_fraction_of_gpu_memory_to_use=xxx`.%s\n\n",
        gpu_id_,
        string::HumanReadableSize(size),
        gpu_id_,
        string::HumanReadableSize(allocated),
        string::HumanReadableSize(avail),
        gpu_id_,
        FLAGS_fraction_of_gpu_memory_to_use,
        err_msg));
  }
}

void GPUAllocator::Free(void* p, size_t size, size_t index) {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    common::errors::InvalidArgument(
                        "The index should be 0, index is %d", index));
  PADDLE_ENFORCE_GE(gpu_alloc_size_,
                    size,
                    common::errors::InvalidArgument(
                        "The size of memory (%d) to free exceeds the size of "
                        "allocated gpu memory (%d)",
                        size,
                        gpu_alloc_size_));
  gpu_alloc_size_ -= size;

  platform::RecordedGpuFree(p, size, gpu_id_);
}

bool GPUAllocator::UseGpu() const { return true; }

// PINNED memory allows direct DMA transfers by the GPU to and from system
// memory. It’s locked to a physical address.
void* CUDAPinnedAllocator::Alloc(size_t* index, size_t size) {
  if (size <= 0) return nullptr;

  // NOTE: here, we use CUDAPinnedMaxAllocSize as the maximum memory size
  // of host pinned allocation. Allocates too much would reduce
  // the amount of memory available to the underlying system for paging.
  size_t usable =
      phi::backends::cpu::CUDAPinnedMaxAllocSize() - cuda_pinnd_alloc_size_;

  if (size > usable) {
    LOG(WARNING) << "Cannot malloc " << size / 1024.0 / 1024.0
                 << " MB pinned memory."
                 << ", available " << usable / 1024.0 / 1024.0
                 << " MB";  // NOLINT
    return nullptr;
  }

  void* p;
// PINNED memory is visible to all CUDA contexts.
#ifdef PADDLE_WITH_HIP
  hipError_t result = hipHostMalloc(&p, size, hipHostMallocPortable);
#else
  cudaError_t result = cudaHostAlloc(&p, size, cudaHostAllocPortable);
#endif

  if (result == gpuSuccess) {
    *index = 1;  // PINNED memory
    cuda_pinnd_alloc_size_ += size;
    HOST_MEMORY_STAT_UPDATE(Reserved, 0, size);
    platform::RecordMemEvent(
        p, CPUPlace(), size, phi::TracerMemEventType::ReservedAllocate);
    return p;
  } else {
    LOG(WARNING) << "cudaHostAlloc failed.";
    return nullptr;
  }

  return nullptr;
}

void CUDAPinnedAllocator::Free(void* p, size_t size, size_t index) {
  gpuError_t err;
  PADDLE_ENFORCE_EQ(index,
                    1,
                    common::errors::InvalidArgument(
                        "The index should be 1, but got %d", index));

  PADDLE_ENFORCE_GE(cuda_pinnd_alloc_size_,
                    size,
                    common::errors::InvalidArgument(
                        "The size of memory (%d) to free exceeds the size of "
                        "allocated cuda pinned memory (%d)",
                        size,
                        cuda_pinnd_alloc_size_));
  cuda_pinnd_alloc_size_ -= size;
#ifdef PADDLE_WITH_HIP
  err = hipHostFree(p);
  if (err != hipErrorDeinitialized) {
    PADDLE_ENFORCE_EQ(
        err,
        hipSuccess,
        common::errors::Fatal(
            "hipFreeHost failed in GPUPinnedAllocator, error code is %d", err));
  }
#else
  err = cudaFreeHost(p);

  // Purposefully allow cudaErrorCudartUnloading, because
  // that is returned if you ever call cudaFreeHost after the
  // driver has already shutdown. This happens only if the
  // process is terminating, in which case we don't care if
  // cudaFreeHost succeeds.
  if (err != cudaErrorCudartUnloading) {
    PADDLE_ENFORCE_EQ(
        err,
        0,
        common::errors::Fatal(
            "cudaFreeHost failed in GPUPinnedAllocator, error code is %d",
            err));
  }
#endif
  HOST_MEMORY_STAT_UPDATE(Reserved, 0, -size);
  platform::RecordMemEvent(
      p, CPUPlace(), size, phi::TracerMemEventType::ReservedFree);
}

bool CUDAPinnedAllocator::UseGpu() const { return false; }

#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
void* CustomAllocator::Alloc(size_t* index, size_t size) {
  if (size <= 0) return nullptr;

  void* p;
  auto place = phi::CustomPlace(dev_type_, dev_id_);
  auto device = phi::DeviceManager::GetDeviceWithPlace(place);
  p = device->MemoryAllocate(size);
  if (LIKELY(p)) {
    VLOG(4) << "CustomAllocator::Alloc " << p << " size " << size;
    *index = 0;
    plug_alloc_size += size;
    DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id_, size);
    platform::RecordMemEvent(
        p, place, size, phi::TracerMemEventType::ReservedAllocate);
  } else {
    size_t avail, total;

    phi::DeviceManager::MemoryStats(place, &total, &avail);
    PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
        "\n\nOut of memory error on %s %d. "
        "total memory is %s, used memory is %s, "
        "available memory is only %s.\n\n",
        dev_type_,
        dev_id_,
        string::HumanReadableSize(total),
        string::HumanReadableSize(total - avail),
        string::HumanReadableSize(avail)));
  }
  return p;
}

void CustomAllocator::Free(void* p, size_t size, size_t index) {
  VLOG(4) << "CustomAllocator::Free " << p << " size " << size;
  PADDLE_ENFORCE_EQ(index,
                    0,
                    common::errors::InvalidArgument(
                        "The index should be 0, index is %d", index));
  PADDLE_ENFORCE_GE(plug_alloc_size,
                    size,
                    common::errors::InvalidArgument(
                        "The size of memory (%d) to free exceeds the size of "
                        "allocated gpu memory (%d)",
                        size,
                        plug_alloc_size));
  plug_alloc_size -= size;
  auto place = phi::CustomPlace(dev_type_, dev_id_);
  auto device = phi::DeviceManager::GetDeviceWithPlace(place);
  device->MemoryDeallocate(p, size);
  DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id_, size);
  platform::RecordMemEvent(
      p, place, size, phi::TracerMemEventType::ReservedFree);
}

bool CustomAllocator::UseGpu() const { return true; }
#endif

}  // namespace paddle::memory::detail
