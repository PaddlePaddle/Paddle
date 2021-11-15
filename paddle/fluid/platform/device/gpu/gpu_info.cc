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

#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include <cstdlib>
#include <mutex>
#include <vector>

#include "gflags/gflags.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/dynload/miopen.h"
#else
#include "paddle/fluid/platform/device/gpu/cuda/cuda_graph.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#endif
#include "paddle/fluid/memory/malloc.h"
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10020
#include "paddle/fluid/platform/dynload/cuda_driver.h"
#endif
#endif
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/lock_guard_ptr.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/monitor.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"

DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_uint64(initial_gpu_memory_in_mb);
DECLARE_uint64(reallocate_gpu_memory_in_mb);
DECLARE_bool(enable_cublas_tensor_op_math);
DECLARE_string(selected_gpus);
DECLARE_uint64(gpu_memory_limit_mb);

constexpr static float fraction_reserve_gpu_memory = 0.05f;

USE_GPU_MEM_STAT;
namespace paddle {
namespace platform {
//! Get a list of device ids from environment variable or use all.
std::vector<int> GetSelectedDevices() {
  // use user specified GPUs in single-node multi-process mode.
  std::vector<int> devices;
  if (!FLAGS_selected_gpus.empty()) {
    auto devices_str = paddle::string::Split(FLAGS_selected_gpus, ',');
    for (auto id : devices_str) {
      devices.push_back(atoi(id.c_str()));
    }
  } else {
    int count = GetGPUDeviceCount();
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  }
  return devices;
}

void GpuMemoryUsage(size_t *available, size_t *total) {
  size_t actual_available, actual_total;
  RecordedGpuMemGetInfo(available, total, &actual_available, &actual_total,
                        platform::GetCurrentDeviceId());
}

size_t GpuAvailableMemToAlloc() {
  size_t total = 0;
  size_t available = 0;
  GpuMemoryUsage(&available, &total);
  size_t reserving =
      static_cast<size_t>(fraction_reserve_gpu_memory * available);
  // If available size is less than minimum chunk size, no usable memory exists
  size_t available_to_alloc = available - reserving;
  size_t min_chunk_size = GpuMinChunkSize();
  if (available_to_alloc < min_chunk_size) {
    available_to_alloc = 0;
  }
  VLOG(10) << "GPU usage " << (available >> 20) << "M/" << (total >> 20)
           << "M, " << (available_to_alloc >> 20) << "M available to allocate";
  return available_to_alloc;
}

size_t GpuMaxAllocSize() {
  return std::max(GpuInitAllocSize(), GpuReallocSize());
}

static size_t GpuAllocSize(bool realloc) {
  size_t available_to_alloc = GpuAvailableMemToAlloc();
  PADDLE_ENFORCE_GT(
      available_to_alloc, 0,
      platform::errors::ResourceExhausted("Not enough available GPU memory."));
  // If FLAGS_initial_gpu_memory_in_mb is 0, then initial memory will be
  // allocated by fraction
  size_t flag_mb = realloc ? FLAGS_reallocate_gpu_memory_in_mb
                           : FLAGS_initial_gpu_memory_in_mb;
  size_t alloc_bytes =
      (flag_mb > 0ul ? flag_mb << 20 : available_to_alloc *
                                           FLAGS_fraction_of_gpu_memory_to_use);
  PADDLE_ENFORCE_GE(
      available_to_alloc, alloc_bytes,
      platform::errors::ResourceExhausted("Not enough available GPU memory."));
  VLOG(10) << "Alloc size is " << (alloc_bytes >> 20)
           << " MiB, is it Re-alloc: " << realloc;
  return alloc_bytes;
}

size_t GpuInitAllocSize() { return GpuAllocSize(/* realloc = */ false); }

size_t GpuReallocSize() { return GpuAllocSize(/* realloc = */ true); }

size_t GpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 256 bytes.
  return 1 << 8;
}

size_t GpuMaxChunkSize() {
  size_t max_chunk_size = GpuMaxAllocSize();
  VLOG(10) << "Max chunk size " << (max_chunk_size >> 20) << "M";
  return max_chunk_size;
}

static void RaiseNonOutOfMemoryError(gpuError_t *status) {
#ifdef PADDLE_WITH_HIP
  if (*status == hipErrorOutOfMemory) {
    *status = hipSuccess;
  }
#else
  if (*status == cudaErrorMemoryAllocation) {
    *status = cudaSuccess;
  }
#endif
  PADDLE_ENFORCE_CUDA_SUCCESS(*status);

#ifdef PADDLE_WITH_HIP
  *status = hipGetLastError();
  if (*status == hipErrorOutOfMemory) {
    *status = hipSuccess;
  }
#else
  *status = cudaGetLastError();
  if (*status == cudaErrorMemoryAllocation) {
    *status = cudaSuccess;
  }
#endif
  PADDLE_ENFORCE_CUDA_SUCCESS(*status);
}

class RecordedGpuMallocHelper {
 private:
  explicit RecordedGpuMallocHelper(int dev_id, uint64_t limit_size = 0)
      : dev_id_(dev_id), limit_size_(limit_size) {
    if (NeedRecord()) {
      mtx_.reset(new std::mutex());
    }
  }

  DISABLE_COPY_AND_ASSIGN(RecordedGpuMallocHelper);

 public:
  static RecordedGpuMallocHelper *Instance(int dev_id) {
    std::call_once(once_flag_, [] {
      int dev_cnt = GetGPUDeviceCount();
      instances_.reserve(dev_cnt);
      for (int i = 0; i < dev_cnt; ++i) {
        instances_.emplace_back(
            new RecordedGpuMallocHelper(i, FLAGS_gpu_memory_limit_mb << 20));
      }
    });

    PADDLE_ENFORCE_GE(
        dev_id, 0,
        platform::errors::OutOfRange(
            "Device id must be not less than 0, but got %d.", dev_id));
    PADDLE_ENFORCE_LT(
        dev_id, instances_.size(),
        platform::errors::OutOfRange("Device id %d exceeds gpu card number %d.",
                                     dev_id, instances_.size()));
    return instances_[dev_id].get();
  }

  /**
   * Try to allocate `size` gpu memory. Only cudaErrorMemoryAllocation
   * or cudaSuccess would be returned, and the cudaGetLastError() flag
   * would be clear.
   */
  gpuError_t Malloc(void **ptr, size_t size) {
    LockGuardPtr<std::mutex> lock(mtx_);
    if (UNLIKELY(NeedRecord() && cur_size_.load() + size > limit_size_)) {
#ifdef PADDLE_WITH_HIP
      return hipErrorOutOfMemory;
#else
      return cudaErrorMemoryAllocation;
#endif
    }

    CUDADeviceGuard guard(dev_id_);
#ifdef PADDLE_WITH_HIP
    auto result = hipMalloc(ptr, size);
#else
    CUDAGraphCaptureModeGuard capture_mode_guard;
    auto result = cudaMalloc(ptr, size);
#endif
    if (result == gpuSuccess) {
      cur_size_.fetch_add(size);
      STAT_INT_ADD("STAT_gpu" + std::to_string(dev_id_) + "_mem_size", size);
      return gpuSuccess;
    } else {
      RaiseNonOutOfMemoryError(&result);
// Non out of memory error would be raised inside
// RaiseNonOutOfMemoryError. Therefore, we can
// return cudaErrorMemoryAllocation directly here.
#ifdef PADDLE_WITH_HIP
      return hipErrorOutOfMemory;
#else
      return cudaErrorMemoryAllocation;
#endif
    }
  }

  /**
   * Free gpu memory. Usually, free is not allowed to raise error.
   * If it does raise error, the process should be crashed.
   */
  void Free(void *ptr, size_t size) {
    // Purposefully allow cudaErrorCudartUnloading, because
    // that is returned if you ever call cudaFree after the
    // driver has already shutdown. This happens only if the
    // process is terminating, in which case we don't care if
    // cudaFree succeeds.
    CUDADeviceGuard guard(dev_id_);
#ifdef PADDLE_WITH_HIP
    auto err = hipFree(ptr);
    if (err != hipErrorDeinitialized) {
#else
    auto err = cudaFree(ptr);
    if (err != cudaErrorCudartUnloading) {
#endif
      PADDLE_ENFORCE_CUDA_SUCCESS(err);
      cur_size_.fetch_sub(size);
      STAT_INT_SUB("STAT_gpu" + std::to_string(dev_id_) + "_mem_size", size);
    } else {
#ifdef PADDLE_WITH_HIP
      hipGetLastError();  // clear the error flag when hipErrorDeinitialized
#else
      cudaGetLastError();  // clear the error flag when cudaErrorCudartUnloading
#endif
    }
  }

  bool GetMemInfo(size_t *avail, size_t *total, size_t *actual_avail,
                  size_t *actual_total) {
    {
      CUDADeviceGuard guard(dev_id_);
#ifdef PADDLE_WITH_HIP
      auto result = hipMemGetInfo(actual_avail, actual_total);
#else
      auto result = cudaMemGetInfo(actual_avail, actual_total);
#endif
      if (result != gpuSuccess) {
        *actual_avail = 0;
      }
      RaiseNonOutOfMemoryError(&result);
    }

    if (NeedRecord()) {
      std::lock_guard<std::mutex> guard(*mtx_);
      *avail = std::min(*actual_avail, limit_size_ - cur_size_.load());
      *total = std::min(*actual_total, limit_size_);
      return *total < *actual_total;
    } else {
      *avail = *actual_avail;
      *total = *actual_total;
      return false;
    }
  }

  inline bool NeedRecord() const { return limit_size_ != 0; }

  uint64_t RecordedSize() const { return cur_size_.load(); }

  uint64_t LimitSize() const { return limit_size_; }

#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10020
  CUresult MemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop,
                     unsigned long long flags) {  // NOLINT
    auto result =
        paddle::platform::dynload::cuMemCreate(handle, size, prop, flags);
    if (result == CUDA_SUCCESS) {
      cur_size_.fetch_add(size);
    }
    return result;
  }

  CUresult MemRelease(CUmemGenericAllocationHandle handle, size_t size) {
    auto result = paddle::platform::dynload::cuMemRelease(handle);
    if (result == CUDA_SUCCESS) {
      cur_size_.fetch_sub(size);
    }
    return result;
  }

#endif
#endif

 private:
  const int dev_id_;
  const uint64_t limit_size_;
  std::atomic<uint64_t> cur_size_{0};

  mutable std::unique_ptr<std::mutex> mtx_;

  static std::once_flag once_flag_;
  static std::vector<std::unique_ptr<RecordedGpuMallocHelper>> instances_;
};  // NOLINT

std::once_flag RecordedGpuMallocHelper::once_flag_;
std::vector<std::unique_ptr<RecordedGpuMallocHelper>>
    RecordedGpuMallocHelper::instances_;

gpuError_t RecordedGpuMalloc(void **ptr, size_t size, int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->Malloc(ptr, size);
}

void RecordedGpuFree(void *p, size_t size, int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->Free(p, size);
}

#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10020
CUresult RecordedGpuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                              const CUmemAllocationProp *prop,
                              unsigned long long flags, int dev_id) {  // NOLINT
  return RecordedGpuMallocHelper::Instance(dev_id)->MemCreate(handle, size,
                                                              prop, flags);
}

CUresult RecordedGpuMemRelease(CUmemGenericAllocationHandle handle, size_t size,
                               int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->MemRelease(handle, size);
}
#endif
#endif

bool RecordedGpuMemGetInfo(size_t *avail, size_t *total, size_t *actual_avail,
                           size_t *actual_total, int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->GetMemInfo(
      avail, total, actual_avail, actual_total);
}

uint64_t RecordedGpuMallocSize(int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->RecordedSize();
}

bool IsGpuMallocRecorded(int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->NeedRecord();
}

void EmptyCache(void) {
  std::vector<int> devices = GetSelectedDevices();
  for (auto device : devices) {
    memory::Release(CUDAPlace(device));
  }
}

}  // namespace platform
}  // namespace paddle
