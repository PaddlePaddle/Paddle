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

#include "paddle/fluid/platform/cpu_info.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include "gflags/gflags.h"

DEFINE_double(fraction_of_cpu_memory_to_use, 1,
              "Default use 100% of CPU memory for PaddlePaddle,"
              "reserve the rest for page tables, etc");

DEFINE_uint64(initial_cpu_memory_in_mb,
#ifdef PADDLE_WITH_MKLDNN
              /* Aligned with mozga-intel, MKLDNN need at least 5000 MB
               * to obtain the best performance*/
              5000,
#else
              500,
#endif
              "Initial CPU memory for PaddlePaddle, in MD unit.");

DEFINE_double(
    fraction_of_cuda_pinned_memory_to_use, 0.5,
    "Default use 50% of CPU memory as the pinned_memory for PaddlePaddle,"
    "reserve the rest for page tables, etc");

namespace paddle {
namespace platform {

inline size_t CpuTotalPhysicalMemory() {
#ifdef __APPLE__
  int mib[2];
  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;
  int64_t size = 0;
  size_t len = sizeof(size);
  if (sysctl(mib, 2, &size, &len, NULL, 0) == 0) return (size_t)size;
  return 0L;
#else
  int64_t pages = sysconf(_SC_PHYS_PAGES);
  int64_t page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
#endif
}

size_t CpuMaxAllocSize() {
  // For distributed systems, it requires configuring and limiting
  // the fraction of memory to use.
  return FLAGS_fraction_of_cpu_memory_to_use * CpuTotalPhysicalMemory();
}

size_t CpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 4 KB.
  return 1 << 12;
}

size_t CpuMaxChunkSize() {
  // Allow to allocate the maximum chunk size is roughly 3% of CPU memory,
  // or the initial_cpu_memory_in_mb.
  return std::min(
      static_cast<size_t>(CpuMaxAllocSize() / 32),
      static_cast<size_t>(FLAGS_initial_cpu_memory_in_mb * 1 << 20));
}

size_t CUDAPinnedMaxAllocSize() {
  // For distributed systems, it requires configuring and limiting
  // the fraction of memory to use.
  return FLAGS_fraction_of_cuda_pinned_memory_to_use * CpuTotalPhysicalMemory();
}

size_t CUDAPinnedMinChunkSize() {
  // Allow to allocate the minimum chunk size is 64 KB.
  return 1 << 16;
}

size_t CUDAPinnedMaxChunkSize() {
  // Allow to allocate the maximum chunk size is roughly 1/256 of CUDA_PINNED
  // memory.
  return CUDAPinnedMaxAllocSize() / 256;
}

}  // namespace platform
}  // namespace paddle
