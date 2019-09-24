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

#pragma once
#include <map>
#include <memory>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/cuda_device_context_allocator.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

/**
 * CUDADeviceContextAllocatorPool is a singletion stores mapping from
 * CUDAPlace(s) to std::shared_ptr<CUDADeviceContextAllocator>. When a
 * CUDADeviceContext's compute stream isn't default stream, it can call this
 * class to allocate GPU memory which will be released by a callback after
 * stream execution.
 */
class CUDADeviceContextAllocatorPool {
 public:
  static CUDADeviceContextAllocatorPool &Instance();

  AllocationPtr Alloc(const platform::CUDADeviceContext &dev_ctx, size_t size);

 private:
  CUDADeviceContextAllocatorPool();
  std::map<platform::CUDAPlace, std::shared_ptr<CUDADeviceContextAllocator>>
      allocators_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
