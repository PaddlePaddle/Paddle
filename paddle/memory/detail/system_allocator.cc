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
#include "paddle/platform/error.h"
#include "paddle/platform/gpu_info.h"

#include <stdlib.h>    // for malloc and free
#include <sys/mman.h>  // for mlock and munlock

#include "gflags/gflags.h"

// If use_pinned_memory is true, CPUAllocator calls mlock, which
// returns pinned and locked memory as staging areas for data exchange
// between host and device.  Allocates too much would reduce the amount
// of memory available to the system for paging.  So, by default, we
// should set false to use_pinned_memory.
DEFINE_bool(use_pinned_memory, false, "If set, allocate cpu pinned memory.");

namespace paddle {
namespace memory {
namespace detail {

void* CPUAllocator::Alloc(size_t& index, size_t size) {
  // According to http://www.cplusplus.com/reference/cstdlib/malloc/,
  // malloc might not return nullptr if size is zero, but the returned
  // pointer shall not be dereferenced -- so we make it nullptr.
  if (size <= 0) return nullptr;

  if (FLAGS_use_pinned_memory) {
    void* p = malloc(size);
    if (p != nullptr) {
      mlock(p, size);
    }
  }

  void* p = malloc(size);
  if (p != nullptr && FLAGS_use_pinned_memory) {
    mlock(p, size);
  }
  return p;
}

void CPUAllocator::Free(void* p, size_t size, size_t index) {
  if (p != nullptr && FLAGS_use_pinned_memory) {
    munlock(p, size);
  }
  free(p);
}

#ifndef PADDLE_ONLY_CPU

void* GPUAllocator::Alloc(size_t& index, size_t size) {
  // CUDA documentation doesn't explain if cudaMalloc returns nullptr
  // if size is 0.  We just make sure it does.
  if (size <= 0) return nullptr;

  size_t available = 0;
  size_t capacity = 0;
  paddle::platform::GpuMemoryUsage(available, capacity);

  // Reserve memory for page tables, etc.
  size_t reserving = capacity - paddle::platform::GpuMaxAllocSize();
  size_t remaining = available > reserving ? available - reserving : 0;

  // If remaining size no less than expected size, using general
  // cudaMalloc to allocate GPU memory.
  void* p = 0;
  if (size <= remaining) {
    cudaError_t result = cudaMalloc(&p, size);
    if (result == cudaSuccess) {
      index = 0;
      total_alloc_size_ += size;
      return p;
    }
  }

  // If remaining size less than expected size or cudaMalloc failed,
  // cudaMallocHost will be considered as a fallback allocator.
  cudaError_t result = cudaMallocHost(&p, size);
  if (result == cudaSuccess) {
    index = 1;
    total_alloc_size_ += size;
    return p;
  }

  return nullptr;
}

void GPUAllocator::Free(void* p, size_t size, size_t index) {
  // Purposefully allow cudaErrorCudartUnloading, because
  // that is returned if you ever call cudaFree after the
  // driver has already shutdown. This happens only if the
  // process is terminating, in which case we don't care if
  // cudaFree succeeds.
  PADDLE_ASSERT(total_alloc_size_ >= size);
  total_alloc_size_ -= size;
  cudaError_t err = index == 1 ? cudaFreeHost(p) : cudaFree(p);
  if (err != cudaErrorCudartUnloading) {
    platform::throw_on_error(err, "cudaFree{Host} failed");
  }
}

#endif  // PADDLE_ONLY_CPU

}  // namespace detail
}  // namespace memory
}  // namespace paddle
