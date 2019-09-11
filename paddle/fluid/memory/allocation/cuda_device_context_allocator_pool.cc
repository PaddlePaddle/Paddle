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
#include "paddle/fluid/platform/enforce.h"

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
  PADDLE_ENFORCE_EQ(iter != allocators_.end(), true,
                    "CUDADeviceContextAllocatorPool initialization error");
  auto &allocator = iter->second;
  AllocationPtr allocation = allocator->Allocate(size);
  static_cast<CUDADeviceContextAllocation *>(allocation.get())
      ->SetCUDADeviceContext(&dev_ctx);
  return allocation;
}

CUDADeviceContextAllocatorPool::CUDADeviceContextAllocatorPool() {
  std::vector<int> devices = platform::GetSelectedDevices();
  for (int i : devices) {
    auto place = platform::CUDAPlace(i);
    auto compute_stream =
        platform::DeviceContextPool::Instance().GetByPlace(place)->stream();
    auto allocator = std::shared_ptr<CUDADeviceContextAllocator>(
        new CUDADeviceContextAllocator(place, compute_stream));
    allocators_.insert(make_pair(place, allocator));
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
