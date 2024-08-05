// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/cpu_allocator.h"

#include <cstdlib>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/stats.h"

namespace paddle::memory::allocation {

bool CPUAllocator::IsAllocThreadSafe() const { return true; }

void CPUAllocator::FreeImpl(phi::Allocation *allocation) {
  auto size = allocation->size();
  void *p = allocation->ptr();
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);  // NOLINT
#endif
  HOST_MEMORY_STAT_UPDATE(Reserved, 0, -size);
  delete allocation;
}

phi::Allocation *CPUAllocator::AllocateImpl(size_t size) {
  void *p = nullptr;
#ifdef _WIN32
  p = _aligned_malloc(size, kAlignment);
#else
  int error = posix_memalign(&p, kAlignment, size);
  PADDLE_ENFORCE_EQ(
      error,
      0,
      common::errors::ResourceExhausted(
          "Fail to alloc memory of %ld size, error code is %d.", size, error));
#endif
  HOST_MEMORY_STAT_UPDATE(Reserved, 0, size);
  return new Allocation(p, size, phi::CPUPlace());
}
}  // namespace paddle::memory::allocation
