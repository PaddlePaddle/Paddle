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

#include <cuda.h>
#include <cuda_runtime.h>

namespace paddle {
namespace memory {
namespace allocation {
bool CPUPinnedAllocator::IsAllocThreadSafe() const { return true; }
void CPUPinnedAllocator::FreeImpl(Allocation *allocation) {
  cudaError_t error = cudaFreeHost(allocation->ptr());
  PADDLE_ENFORCE_EQ(error, 0,
                    platform::errors::ResourceExhausted(
                        "Free memory allocation failed using cudaFreeHost, "
                        "error code is %d",
                        error));
  delete allocation;
}
Allocation *CPUPinnedAllocator::AllocateImpl(size_t size) {
  void *ptr;
  cudaError_t error = cudaHostAlloc(&ptr, size, cudaHostAllocPortable);
  PADDLE_ENFORCE_EQ(
      error, 0,
      platform::errors::ResourceExhausted(
          "Alloc memory allocation of size %d failed using cudaHostAlloc, "
          "error code is %d",
          size, error));
  return new Allocation(ptr, size, platform::CUDAPinnedPlace());
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
