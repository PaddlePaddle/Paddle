// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/cuda_device_context_allocator_pool.h"

#include <utility>
#include <vector>
#include "paddle/fluid/memory/allocation/cuda_device_context_allocation.h"
#include "paddle/fluid/memory/allocation/cuda_device_context_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

CUDADeviceContextAllocatorPool &CUDADeviceContextAllocatorPool::Instance() {
  static CUDADeviceContextAllocatorPool pool;
  return pool;
}

AllocationPtr CUDADeviceContextAllocatorPool::Alloc(
    const platform::CUDADeviceContext &dev_ctx, size_t size) {
  auto iter =
      allocators_.find(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()));
  PADDLE_ENFORCE(iter != allocators_.end());
  auto allocation = iter->second->Allocate(size);
  static_cast<CUDADeviceContextAllocation *>(allocation.get())
      ->SetCUDADeviceContext(&dev_ctx);
  return allocation;
}

CUDADeviceContextAllocatorPool::CUDADeviceContextAllocatorPool() {
  std::vector<int> devices = platform::GetSelectedDevices();
  for (int i : devices) {
    auto place = platform::CUDAPlace(i);
    allocators_.insert(
        make_pair(place, std::shared_ptr<CUDADeviceContextAllocator>(
                             new CUDADeviceContextAllocator(place))));
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
