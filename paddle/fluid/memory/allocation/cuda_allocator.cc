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

#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {
bool CUDAAllocator::IsAllocThreadSafe() const { return true; }
void CUDAAllocator::FreeImpl(Allocation* allocation) {
  platform::CUDADeviceGuard guard(place_.device);
  PADDLE_ENFORCE_EQ(boost::get<platform::CUDAPlace>(allocation->place()),
                    place_);
  PADDLE_ENFORCE(cudaFree(allocation->ptr()));
  delete allocation;
}

Allocation* CUDAAllocator::AllocateImpl(size_t size, Allocator::Attr attr) {
  platform::CUDADeviceGuard guard(place_.device);
  void* ptr;
  auto status = cudaMalloc(&ptr, size);
  if (UNLIKELY(status != cudaSuccess)) {
    throw BadAlloc(string::Sprintf(
        "Cannot allocate %d on GPU %d, cuda status %d, %s", size, place_.device,
        status, cudaGetErrorString(status)));
  }
  return new Allocation(ptr, size, platform::Place(place_));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
