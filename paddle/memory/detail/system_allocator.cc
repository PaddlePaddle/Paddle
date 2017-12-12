/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/memory/detail/system_allocator.h"
#include "paddle/platform/assert.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/gpu_info.h"

#include <stdlib.h>    // for malloc and free
#include <sys/mman.h>  // for mlock and munlock

#include "gflags/gflags.h"

// If use_pinned_memory is true, CPUAllocator calls mlock, which
// returns pinned and locked memory as staging areas for data exchange
// between host and device.  Allocates too much would reduce the amount
// of memory available to the system for paging.  So, by default, we
// should set false to use_pinned_memory.
DEFINE_bool(use_pinned_memory, true, "If set, allocate cpu pinned memory.");

namespace paddle {
namespace memory {
namespace detail {

void* CPUAllocator::Alloc(size_t& index, size_t size) {
  // According to http://www.cplusplus.com/reference/cstdlib/malloc/,
  // malloc might not return nullptr if size is zero, but the returned
  // pointer shall not be dereferenced -- so we make it nullptr.
  if (size <= 0) return nullptr;

  index = 0;  // unlock memory

  void* p;

#ifdef PADDLE_WITH_MKLDNN
  // refer to https://github.com/01org/mkl-dnn/blob/master/include/mkldnn.hpp
  // memory alignment
  PADDLE_ENFORCE_EQ(posix_memalign(&p, 4096ul, size), 0);
#else
  PADDLE_ENFORCE_EQ(posix_memalign(&p, 32ul, size), 0);
#endif
  PADDLE_ENFORCE(p, "Fail to allocate CPU memory: size = %d .", size);

  if (p != nullptr) {
    if (FLAGS_use_pinned_memory) {
      index = 1;
      mlock(p, size);  // lock memory
    }
  }

  return p;
}

void CPUAllocator::Free(void* p, size_t size, size_t index) {
  if (p != nullptr && index == 1) {
    munlock(p, size);
  }
  free(p);
}

bool CPUAllocator::UseGpu() const { return false; }

#ifdef PADDLE_WITH_CUDA

void* GPUAllocator::Alloc(size_t& index, size_t size) {
  // CUDA documentation doesn't explain if hipMalloc returns nullptr
  // if size is 0.  We just make sure it does.
  if (size <= 0) return nullptr;

  size_t available = 0;
  size_t capacity = 0;
  paddle::platform::GpuMemoryUsage(available, capacity);

  // Reserve memory for page tables, etc.
  size_t reserving = 0.05 * capacity + paddle::platform::GpuMinChunkSize();
  size_t usable = available > reserving ? available - reserving : 0;

  // If remaining size no less than expected size, using general
  // hipMalloc to allocate GPU memory.
  void* p = 0;
  if (size <= usable) {
    hipError_t result = hipMalloc(&p, size);
    if (result == hipSuccess) {
      index = 0;
      gpu_alloc_size_ += size;
      return p;
    }
  }

  // If remaining size less than expected size or hipMalloc failed,
  // hipHostMalloc will be considered as a fallback allocator.
  //
  // NOTE: here, we use GpuMaxAllocSize() as the maximum memory size
  // of host fallback allocation. Allocates too much would reduce
  // the amount of memory available to the underlying system for paging.
  usable = paddle::platform::GpuMaxAllocSize() - fallback_alloc_size_;

  if (size > usable) return nullptr;

  hipError_t result = hipHostMalloc(&p, size);
  if (result == hipSuccess) {
    index = 1;
    fallback_alloc_size_ += size;
    return p;
  }

  return nullptr;
}

void GPUAllocator::Free(void* p, size_t size, size_t index) {
  hipError_t err;

  if (index == 0) {
    PADDLE_ASSERT(gpu_alloc_size_ >= size);
    gpu_alloc_size_ -= size;
    err = hipFree(p);
  } else {
    PADDLE_ASSERT(fallback_alloc_size_ >= size);
    fallback_alloc_size_ -= size;
    err = hipHostFree(p);
  }

  // Purposefully allow cudaErrorCudartUnloading, because
  // that is returned if you ever call hipFree after the
  // driver has already shutdown. This happens only if the
  // process is terminating, in which case we don't care if
  // hipFree succeeds.
#if 0
  if (err != cudaErrorCudartUnloading) {
    PADDLE_ENFORCE(err, "hipFree{Host} failed in GPUAllocator::Free.");
  }
#else
  if (err != hipSuccess) {
    err = hipSuccess;
  }
#endif
}

bool GPUAllocator::UseGpu() const { return true; }

#endif

}  // namespace detail
}  // namespace memory
}  // namespace paddle
