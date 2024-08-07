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

#include "paddle/fluid/framework/new_executor/workqueue/workqueue_utils.h"

#include <cstdint>
#include <cstdlib>

namespace paddle::framework {

void* AlignedMalloc(size_t size, size_t alignment) {
  assert(alignment >= sizeof(void*) && (alignment & (alignment - 1)) == 0);
  size = (size + alignment - 1) / alignment * alignment;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
  void* aligned_mem = nullptr;
  if (posix_memalign(&aligned_mem, alignment, size) != 0) {
    aligned_mem = nullptr;
  }
  return aligned_mem;
#elif defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void* mem = malloc(size + alignment);  // NOLINT
  if (mem == nullptr) {
    return nullptr;
  }
  size_t adjust = alignment - reinterpret_cast<uint64_t>(mem) % alignment;
  void* aligned_mem = reinterpret_cast<char*>(mem) + adjust;
  *(reinterpret_cast<void**>(aligned_mem) - 1) = mem;
  assert(reinterpret_cast<uint64_t>(aligned_mem) % alignment == 0);
  return aligned_mem;
#endif
}

void AlignedFree(void* mem_ptr) {
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
  free(mem_ptr);
#elif defined(_WIN32)
  _aligned_free(mem_ptr);
#else
  if (mem_ptr) {
    free(*(reinterpret_cast<void**>(mem_ptr) - 1));  // NOLINT
  }
#endif
}

}  // namespace paddle::framework
