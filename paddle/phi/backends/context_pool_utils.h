/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/backends/all_context.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/generator.h"

namespace phi {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename DevCtx>
typename std::enable_if<!std::is_same<DevCtx, phi::GPUContext>::value,
                        DevCtx*>::type
ConstructDevCtx(const phi::Place& p, /*unused*/ int stream_priority = 0) {
  return new DevCtx(p);
}

template <typename DevCtx>
typename std::enable_if<std::is_same<DevCtx, phi::GPUContext>::value,
                        DevCtx*>::type
ConstructDevCtx(const phi::Place& p, int stream_priority) {
  return new DevCtx(p, /*init=*/true, stream_priority);
}
#else
template <typename DevCtx>
DevCtx* ConstructDevCtx(const phi::Place& p,
                        /*unused*/ int stream_priority) {
  return new DevCtx(p);
}
#endif

template <typename DevCtx>
inline std::unique_ptr<DeviceContext> CreateDeviceContext(
    const phi::Place& p,
    bool disable_setting_default_stream_for_allocator,
    int stream_priority) {
  using PtrType = std::unique_ptr<DeviceContext>;

  DevCtx* dev_ctx = ConstructDevCtx<DevCtx>(p, stream_priority);

  if (p.GetType() == phi::AllocationType::GPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto* cuda_ctx = dynamic_cast<phi::GPUContext*>(dev_ctx);
    PADDLE_ENFORCE_NOT_NULL(
        cuda_ctx,
        phi::errors::InvalidArgument(
            "Failed to dynamic_cast dev_ctx into phi::GPUContext."));

    auto& instance = paddle::memory::allocation::AllocatorFacade::Instance();
    if (!disable_setting_default_stream_for_allocator) {
      instance.SetDefaultStream(GPUPlace(p.GetDeviceId()), cuda_ctx->stream());
    }
    dev_ctx->SetAllocator(instance.GetAllocator(p, cuda_ctx->stream()).get());
    dev_ctx->SetPinnedAllocator(
        instance.GetAllocator(phi::GPUPinnedPlace()).get());

    cuda_ctx->PartialInitWithAllocator();
    dev_ctx->SetGenerator(phi::DefaultCUDAGenerator(p.GetDeviceId()).get());
#endif
  } else if (p.GetType() == phi::AllocationType::XPU) {
#if defined(PADDLE_WITH_XPU)
    dev_ctx->SetAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(p)
            .get());
    dev_ctx->SetGenerator(phi::DefaultXPUGenerator(p.GetDeviceId()).get());
#endif
  } else {
    dev_ctx->SetAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(p)
            .get());
    dev_ctx->SetGenerator(phi::DefaultCPUGenerator().get());
  }
  dev_ctx->SetHostGenerator(phi::DefaultCPUGenerator().get());
  dev_ctx->SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  dev_ctx->SetZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(p)
          .get());
  dev_ctx->SetHostZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(phi::CPUPlace())
          .get());
  return PtrType(dev_ctx);
}

template <typename DevCtx>
inline void EmplaceDeviceContext(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    const phi::Place& place,
    bool disable_setting_default_stream_for_allocator,
    int stream_priority) {
  // lazy evaluation. i.e., only create device context at first `Get`
  place_to_device_context->emplace(
      place,
      std::async(std::launch::deferred,
                 CreateDeviceContext<DevCtx>,
                 place,
                 disable_setting_default_stream_for_allocator,
                 stream_priority));
}

}  // namespace phi
