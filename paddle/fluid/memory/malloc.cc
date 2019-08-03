/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/malloc.h"
#include <string>
#include <vector>
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/memory/allocation/cuda_device_context_allocator_pool.h"
#endif
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {

std::shared_ptr<Allocation> AllocShared(const platform::Place &place,
                                        size_t size) {
  return allocation::AllocatorFacade::Instance().AllocShared(place, size);
}

AllocationPtr Alloc(const platform::Place &place, size_t size) {
  return allocation::AllocatorFacade::Instance().Alloc(place, size);
}

AllocationPtr Alloc(const platform::DeviceContext &dev_ctx, size_t size) {
  auto place = dev_ctx.GetPlace();
#ifdef PADDLE_WITH_CUDA
  if (size == 0 || !platform::is_gpu_place(place)) {
    return Alloc(place, size);
  }
  auto *default_dev_ctx = static_cast<platform::CUDADeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place));
  auto &desired_dev_ctx =
      static_cast<const platform::CUDADeviceContext &>(dev_ctx);
  if (default_dev_ctx->stream() == desired_dev_ctx.stream()) {
    return Alloc(place, size);
  } else {
    return allocation::CUDADeviceContextAllocatorPool::Instance().Alloc(
        desired_dev_ctx, size);
  }
#else
  return Alloc(place, size);
#endif
}

}  // namespace memory
}  // namespace paddle
