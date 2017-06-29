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

#include <stdlib.h>    // for malloc and free
#include <sys/mman.h>  // for mlock and munlock

#include "gflags/gflags.h"
#include "paddle/platform/assert.h"
#include "paddle/platform/cuda.h"

// If use_pinned_memory is true, CPUAllocator calls mlock, which
// returns pinned and locked memory as staging areas for data exchange
// between host and device.  Allocates too much would reduce the amount
// of memory available to the system for paging.  So, by default, we
// should set false to use_pinned_memory.
DEFINE_bool(use_pinned_memory, false,
            "If set, allocate cpu/gpu pinned memory.");

namespace paddle {
namespace memory {
namespace detail {

void* CPUAllocator::Alloc(size_t size) {
  // According to http://www.cplusplus.com/reference/cstdlib/malloc/,
  // malloc might not return nullptr if size is zero, but the returned
  // pointer shall not be dereferenced -- so we make it nullptr.
  if (size <= 0) return nullptr;

  void* p = malloc(size);
  if (p != nullptr && FLAGS_use_pinned_memory) {
    mlock(p, size);
  }
  return p;
}

void CPUAllocator::Free(void* p, size_t size) {
  if (p != nullptr && FLAGS_use_pinned_memory) {
    munlock(p, size);
  }
  free(p);
}

#ifndef PADDLE_ONLY_CPU

void* GPUAllocator::Alloc(size_t size) {
  // CUDA documentation doesn't explain if cudaMalloc returns nullptr
  // if size is 0.  We just make sure it does.
  if (size <= 0) {
    return nullptr;
  }

  void* p = 0;
  cudaError_t result =
      FLAGS_use_pinned_memory ? cudaMallocHost(&p, size) : cudaMalloc(&p, size);
  if (result != cudaSuccess) {
    cudaGetLastError();  // clear error if there is any.
  }
  return result == cudaSuccess ? p : nullptr;
}

void GPUAllocator::Free(void* p, size_t size) {
  // Purposefully allow cudaErrorCudartUnloading, because
  // that is returned if you ever call cudaFree after the
  // driver has already shutdown. This happens only if the
  // process is terminating, in which case we don't care if
  // cudaFree succeeds.
  cudaError_t err = FLAGS_use_pinned_memory ? cudaFreeHost(p) : cudaFree(p);
  if (err != cudaErrorCudartUnloading) {
    platform::throw_on_error(err, "cudaFree{Host} failed");
  }
}

#endif  // PADDLE_ONLY_CPU

}  // namespace detail
}  // namespace memory
}  // namespace paddle
