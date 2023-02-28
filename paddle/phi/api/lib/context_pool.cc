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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/cuda_stream.h"
#endif

#include "paddle/fluid/platform/init.h"

namespace paddle {
namespace experimental {

DeviceContextPool& DeviceContextPool::Instance() {
  static DeviceContextPool g_device_context_pool;
  return g_device_context_pool;
}

const phi::DeviceContext* DeviceContextPool::Get(const Place& place) {
  auto it = context_map_.find(place);
  if (it == context_map_.end()) {
    if (!paddle::platform::DeviceContextPool::IsInitialized()) {
      paddle::framework::InitDevices();
    }
    // only when we need the specific DeviceContext, get and cache it
    auto* dev_ctx = paddle::platform::DeviceContextPool::Instance().Get(place);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      context_map_[place] = dev_ctx;
    }
    return dev_ctx;
  }
  return it->second;
}

phi::DeviceContext* DeviceContextPool::GetMutable(const Place& place) {
  return const_cast<phi::DeviceContext*>(Get(place));
}

}  // namespace experimental
}  // namespace paddle

namespace paddle {

PADDLE_API phi::Allocator* GetAllocator(const phi::Place& place) {
  const phi::DeviceContext* dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(place);
  return const_cast<phi::Allocator*>(&dev_ctx->GetAllocator());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PADDLE_API phi::CUDAStream* GetCurrentCUDAStream(const phi::Place& place) {
  PADDLE_ENFORCE(place.GetType() == phi::AllocationType::GPU,
                 phi::errors::InvalidArgument(
                     "getCurrentCUDAStream only supports GPUPlace input. "
                     "However, your input is place=%s",
                     place));

  auto& pool = paddle::experimental::DeviceContextPool::Instance();
  const phi::GPUContext* dev_ctx =
      static_cast<const phi::GPUContext*>(pool.Get(place));
  return dev_ctx->cuda_stream();
}
#endif

}  // namespace paddle
