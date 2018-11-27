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
#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif
#include <string>
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {
bool CUDAAllocator::IsAllocThreadSafe() const { return true; }
void CUDAAllocator::Free(Allocation* allocation) {
  platform::CUDADeviceGuard guard(place_.device);
  auto* cuda_allocation = dynamic_cast<CUDAAllocation*>(allocation);
  PADDLE_ENFORCE_NOT_NULL(cuda_allocation);
  PADDLE_ENFORCE_EQ(boost::get<platform::CUDAPlace>(cuda_allocation->place()),
                    place_);
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE(cudaFree(allocation->ptr()));
#endif
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE(hipFree(allocation->ptr()));
#endif
  delete allocation;
}
Allocation* CUDAAllocator::AllocateImpl(size_t size, Allocator::Attr attr) {
  platform::CUDADeviceGuard guard(place_.device);
  void* ptr;
#ifdef PADDLE_WITH_CUDA
  auto status = cudaMalloc(&ptr, size);
  if (UNLIKELY(status != cudaSuccess)) {
    throw BadAlloc(string::Sprintf(
        "Cannot allocate %d on GPU %d, cuda status %d, %s", size, place_.device,
        status, cudaGetErrorString(status)));
  }
#endif
#ifdef PADDLE_WITH_HIP
  auto status = hipMalloc(&ptr, size);
  if (UNLIKELY(status != hipSuccess)) {
    throw BadAlloc(string::Sprintf(
        "Cannot allocate %d on GPU %d, hip status %d, %s", size, place_.device,
        status, hipGetErrorString(status)));
  }
#endif
  return new CUDAAllocation(ptr, size, platform::Place(place_));
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
