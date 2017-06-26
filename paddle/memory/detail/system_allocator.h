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

#pragma once

#include <stddef.h>    // for size_t
#include <sys/mman.h>  // for mlock and munlock
#include <cstdlib>     // for malloc and free

#ifndef PADDLE_ONLY_CPU
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#endif  // PADDLE_ONLY_CPU

namespace paddle {
namespace memory {
namespace detail {

class SystemAllocator {
 public:
  virtual void* Alloc(size_t size) = 0;
  virtual void* Free(void* p) = 0;
};

// CPUAllocator<lock_memory=true> calls mlock, which returns pinned
// and locked memory as staging areas for data exchange between host
// and device.  Allocates too much would reduce the amount of memory
// available to the system for paging.  So, by default, we should use
// CPUAllocator<staging=false>.
template <bool lock_memory>
class CPUAllocator : public SystemAllocator {
 public:
  virtual void* Alloc(size_t size) {
    void* p = std::malloc(size);
    if (p != nullptr && lock_memory) {
      mlock(p, size);
    }
    return p;
  }

  virtual void Free(void* p, size_t size) {
    if (p != nullptr && lock_memory) {
      munlock(p, size);
    }
    std::free(p);
  }
};

#ifndef PADDLE_ONLY_CPU  // The following code are for CUDA.

namespace {
inline void throw_on_error(cudaError_t e, const char* message) {
  if (e) {
    throw thrust::system_error(e, thrust::cuda_category(), message);
  }
}
}  // namespace

// GPUAllocator<staging=true> calls cudaHostMalloc, which returns
// pinned and locked memory as staging areas for data exchange
// between host and device.  Allocates too much would reduce the
// amount of memory available to the system for paging.  So, by
// default, we should use GPUAllocator<staging=false>.
template <bool staging>
class GPUAllocator {
 public:
  void* Alloc(size_t size) {
    void* p = 0;
    cudaError_t result =
        staging ? cudaMallocHost(&p, size) : cudaMalloc(&p, size);
    if (result == cudaSuccess) {
      return p;
    }
    // clear last error
    cudaGetLastError();
    return nullptr;
  }

  void Free(void* p, size_t size) {
    // Purposefully allow cudaErrorCudartUnloading, because
    // that is returned if you ever call cudaFree after the
    // driver has already shutdown. This happens only if the
    // process is terminating, in which case we don't care if
    // cudaFree succeeds.
    auto err = staging ? cudaFreeHost(p) : cudaFree(p);
    if (err != cudaErrorCudartUnloading) {
      throw_on_error(err, "cudaFree failed");
    }
  }
};

#endif  // PADDLE_ONLY_CPU

}  // namespace detail
}  // namespace memory
}  // namespace paddle
