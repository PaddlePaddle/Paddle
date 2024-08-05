/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/api/include/context_pool.h"

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/cuda_stream.h"
#endif

namespace paddle::experimental {

void DeviceContextPool::SyncDeviceContext(const Place& place) {
  if (!phi::DeviceContextPool::IsInitialized()) {
    phi::memory_utils::InitDevices();
  }
  // only when we need the specific DeviceContext, get and cache it
  auto* dev_ctx = phi::DeviceContextPool::Instance().Get(place);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    context_map_[place] = dev_ctx;
  }
}

DeviceContextPool& DeviceContextPool::Instance() {
  static DeviceContextPool g_device_context_pool;
  return g_device_context_pool;
}

const phi::DeviceContext* DeviceContextPool::Get(const Place& place) {
  auto it = context_map_.find(place);
  if (it == context_map_.end()) {
    if (!phi::DeviceContextPool::IsInitialized()) {
      phi::memory_utils::InitDevices();
    }
    // only when we need the specific DeviceContext, get and cache it
    auto* dev_ctx = phi::DeviceContextPool::Instance().Get(place);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      context_map_[place] = dev_ctx;
    }
    return dev_ctx;
  }
  return it->second;
}

phi::DeviceContext* DeviceContextPool::GetMutable(const Place& place) {
  return const_cast<phi::DeviceContext*>(Get(place));  // NOLINT
}

}  // namespace paddle::experimental

namespace paddle {

PADDLE_API phi::Allocator* GetAllocator(const phi::Place& place) {
  const phi::DeviceContext* dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(place);
  return const_cast<phi::Allocator*>(&dev_ctx->GetAllocator());  // NOLINT
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PADDLE_API phi::CUDAStream* GetCurrentCUDAStream(const phi::Place& place) {
  PADDLE_ENFORCE_EQ(place.GetType(),
                    phi::AllocationType::GPU,
                    common::errors::InvalidArgument(
                        "GetCurrentCUDAStream only supports GPUPlace input. "
                        "However, your input is place=%s",
                        place));

  auto& pool = paddle::experimental::DeviceContextPool::Instance();
  const phi::GPUContext* dev_ctx =
      static_cast<const phi::GPUContext*>(pool.Get(place));
  return dev_ctx->cuda_stream();
}
#endif

}  // namespace paddle
