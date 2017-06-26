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

#include <stddef.h>  // for size_t
#include <cstdlib>   // for malloc and free

#ifndef _WIN32
#include <sys/mman.h>  // for mlock and munlock
#endif

namespace paddle {
namespace memory {
namespace detail {

// CPUAllocator<staging=true> calls mlock, which returns
// pinned and locked memory as staging areas for data exchange
// between host and device.  Allocates too much would reduce the
// amount of memory available to the system for paging.  So, by
// default, we should use CPUAllocator<staging=false>.
template <bool staging>
class CPUAllocator {
 public:
  void* Alloc(size_t size);
  void Free(void* p, size_t size);
};

template <>
class CPUAllocator<false> {
 public:
  void* Alloc(size_t size) { return std::malloc(size); }
  void Free(void* p, size_t size) { std::free(p); }
};

template <>
class CPUAllocator<true> {
 public:
  void* Alloc(size_t size) {
    void* p = std::malloc(size);
    if (p == nullptr) {
      return p;
    }
#ifndef _WIN32
    mlock(p, size);
#endif
    return p;
  }

  void Free(void* p, size_t size) {
#ifndef _WIN32
    munlock(p, size);
#endif
    std::free(p);
  }
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
