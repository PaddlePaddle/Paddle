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

#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include <stdlib.h>
#include <string>

namespace paddle {
namespace memory {
namespace allocation {

CPUAllocation::CPUAllocation(void *ptr, size_t size)
    : Allocation(ptr, size, platform::CPUPlace()) {}

void CPUAllocation::share_data_with(void* ptr, size_t size) {
  ptr_ = ptr;
  size_ = size;

  VLOG(10) << "CPUAllocation shares data with ptr: " << ptr << " size: " << size;
}

bool CPUAllocator::IsAllocThreadSafe() const { return true; }

void CPUAllocator::Free(Allocation *allocation) {
  PADDLE_ENFORCE_NOT_NULL(dynamic_cast<CPUAllocation *>(allocation));
  free(allocation->ptr());
  delete allocation;
}

Allocation *CPUAllocator::AllocateImpl(size_t size, Allocator::Attr attr) {
  void *ptr;

  switch (attr) {
    case kNumpyShared:
      ptr = nullptr;
      size = 0U;
      break;
    default:
      auto status = posix_memalign(&ptr, kAlignment, size);
      if (UNLIKELY(status) != 0) {
        throw BadAlloc(string::Sprintf("Cannot allocate cpu memory %d. Errno is %d",
                                       size, status));
      }
      break;
  }

  return new CPUAllocation(ptr, size);
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
