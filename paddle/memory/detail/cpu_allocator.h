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

#include <malloc.h>  // for malloc and free
#include <stddef.h>  // for size_t

#ifdef PADDLE_WITH_GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif  // PADDLE_WITH_GPU

namespace paddle {
namespace memory {
namespace detail {

// CPUAllocator<staging=true> calls cudaMallocHost, which returns
// pinned and mlocked memory as staging areas for data exchange
// between host and device.  Allocates too much would reduce the
// amount of memory available to the system for paging.  So, by
// default, we should use CPUAllocator<staging=false>.
template <bool staging>
class CPUAllocator {
public:
  void* Alloc(size_t size);
  void Free(void* p);
};

template <>
class CPUAllocator<false> {
public:
  void* Alloc(size_t size) { return malloc(size); }
  void Free(void* p) { free(p); }
};

// If CMake macro PADDLE_WITH_GPU is OFF, C++ compiler won't generate the
// following specialization that depends on the CUDA library.
#ifdef PADDLE_WITH_GPU
template <>
class CPUAllocator<true> {
public:
  void* Alloc(size_t size) {
    void* p;
    if (cudaMallocHost(&p, size) != cudaSuccess) {
      return NULL;
    }
    return p;
  }

  void Free(void* p) { cudaFreeHost(p); }
};
#endif  // PADDLE_WITH_GPU

}  // namespace detail
}  // namespace memory
}  // namespace paddle
