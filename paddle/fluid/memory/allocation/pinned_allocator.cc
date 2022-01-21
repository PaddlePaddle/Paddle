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

#include "paddle/fluid/memory/allocation/pinned_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {
bool CPUPinnedAllocator::IsAllocThreadSafe() const { return true; }
void CPUPinnedAllocator::FreeImpl(pten::Allocation *allocation) {
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipHostFree(allocation->ptr()));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaFreeHost(allocation->ptr()));
#endif
  delete allocation;
}
pten::Allocation *CPUPinnedAllocator::AllocateImpl(size_t size) {
  void *ptr;
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipHostMalloc(&ptr, size, hipHostMallocPortable));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaHostAlloc(&ptr, size, cudaHostAllocPortable));
#endif
  return new Allocation(ptr, size, platform::CUDAPinnedPlace());
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
