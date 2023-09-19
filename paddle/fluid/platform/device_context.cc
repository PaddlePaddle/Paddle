/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Corporation. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device_context.h"

#include <functional>
#include <memory>
#include <set>

#include "glog/logging.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/expect.h"
#include "paddle/phi/core/generator.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/memory/allocation/cuda_device_context_allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif

namespace paddle {
namespace platform {

DeviceType Place2DeviceType(const platform::Place& place) {
  if (platform::is_cpu_place(place)) {
    return platform::DeviceType::CPU;
  } else if (platform::is_gpu_place(place)) {
    return platform::DeviceType::CUDA;
  } else if (platform::is_xpu_place(place)) {
    return platform::DeviceType::XPU;
  } else if (platform::is_ipu_place(place)) {
    return platform::DeviceType::IPU;
  } else if (platform::is_custom_place(place)) {
    return platform::DeviceType::CUSTOM_DEVICE;
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported place %s to convert into platform::DeviceType.", place));
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename DevCtx>
typename std::enable_if<!std::is_same<DevCtx, phi::GPUContext>::value,
                        DevCtx*>::type
ConstructDevCtx(const phi::Place& p,
                /*unused*/ int stream_priority UNUSED = 0) {
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
  auto& instance = paddle::memory::allocation::AllocatorFacade::Instance();
  if (p.GetType() == phi::AllocationType::GPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto* cuda_ctx = dynamic_cast<phi::GPUContext*>(dev_ctx);
    PADDLE_ENFORCE_NOT_NULL(
        cuda_ctx,
        phi::errors::InvalidArgument(
            "Failed to dynamic_cast dev_ctx into phi::GPUContext."));

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
    dev_ctx->SetAllocator(instance.GetAllocator(p).get());
    dev_ctx->SetGenerator(phi::DefaultXPUGenerator(p.GetDeviceId()).get());
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (p.GetType() == phi::AllocationType::CUSTOM) {
    auto* custom_ctx = dynamic_cast<phi::CustomContext*>(dev_ctx);
    PADDLE_ENFORCE_NOT_NULL(
        custom_ctx,
        phi::errors::InvalidArgument(
            "Failed to dynamic_cast dev_ctx into phi::CustomContext."));

    if (!disable_setting_default_stream_for_allocator) {
      instance.SetDefaultStream(CustomPlace(p.GetDeviceType(), p.GetDeviceId()),
                                custom_ctx->stream());
    }
    dev_ctx->SetAllocator(instance.GetAllocator(p, custom_ctx->stream()).get());
    dev_ctx->SetGenerator(phi::DefaultCustomDeviceGenerator(p).get());
#endif
  } else {
    dev_ctx->SetAllocator(instance.GetAllocator(p).get());
    dev_ctx->SetGenerator(phi::DefaultCPUGenerator().get());
  }
  dev_ctx->SetHostGenerator(phi::DefaultCPUGenerator().get());
  dev_ctx->SetHostAllocator(instance.GetAllocator(phi::CPUPlace()).get());
  dev_ctx->SetZeroAllocator(instance.GetZeroAllocator(p).get());
  dev_ctx->SetHostZeroAllocator(
      instance.GetZeroAllocator(phi::CPUPlace()).get());
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

void EmplaceDeviceContexts(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    const std::vector<phi::Place>& places,
    bool disable_setting_default_stream_for_allocator,
    int stream_priority) {
  PADDLE_ENFORCE_GT(
      places.size(),
      0,
      phi::errors::InvalidArgument("The number of platform places should "
                                   "be larger than 0. But received %d.",
                                   places.size()));
  std::set<Place> set;
  for (auto& p : places) {
    set.insert(p);
  }
  for (auto& place : set) {
    if (place.GetType() == phi::AllocationType::CPU) {
#ifdef PADDLE_WITH_DNNL
      EmplaceDeviceContext<phi::OneDNNContext>(
          place_to_device_context,
          place,
          disable_setting_default_stream_for_allocator,
          /*unused*/ stream_priority);
#else
      EmplaceDeviceContext<phi::CPUContext>(
          place_to_device_context,
          place,
          disable_setting_default_stream_for_allocator,
          /*unused*/ stream_priority);
#endif
    } else if (place.GetType() == phi::AllocationType::GPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      EmplaceDeviceContext<phi::GPUContext>(
          place_to_device_context,
          place,
          disable_setting_default_stream_for_allocator,
          stream_priority);
#else
      PADDLE_THROW(
          phi::errors::Unimplemented("GPUPlace is not supported. Please "
                                     "re-compile with WITH_GPU option."));
#endif
    } else if (place.GetType() == phi::AllocationType::XPU) {
#ifdef PADDLE_WITH_XPU
      EmplaceDeviceContext<phi::XPUContext>(
          place_to_device_context,
          place,
          disable_setting_default_stream_for_allocator,
          /*unused*/ stream_priority);
#else
      PADDLE_THROW(
          phi::errors::Unimplemented("XPUPlace is not supported. Please "
                                     "re-compile with WITH_XPU option."));
#endif
    } else if (place.GetType() == phi::AllocationType::CUSTOM) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      EmplaceDeviceContext<phi::CustomContext>(
          place_to_device_context,
          place,
          disable_setting_default_stream_for_allocator,
          /*unused*/ stream_priority);
#else
      PADDLE_THROW(phi::errors::Unimplemented(
          "CustomPlace is not supported. Please re-compile with "
          "WITH_CUSTOM_DEVICE "
          "option."));
#endif
    } else if (platform::is_cuda_pinned_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      EmplaceDeviceContext<CUDAPinnedDeviceContext>(
          place_to_device_context,
          place,
          disable_setting_default_stream_for_allocator,
          /*unused*/ stream_priority);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "CUDAPlace is not supported. Please re-compile with WITH_GPU "
          "option."));
#endif
    } else if (platform::is_ipu_place(place)) {
#ifdef PADDLE_WITH_IPU
      EmplaceDeviceContext<IPUDeviceContext>(
          place_to_device_context,
          place,
          disable_setting_default_stream_for_allocator,
          /*unused*/ stream_priority);
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("IPUPlace is not supported. Please "
                                          "re-compile with WITH_IPU option."));
#endif
    }
  }
}

#ifdef PADDLE_WITH_IPU
IPUDeviceContext::IPUDeviceContext(IPUPlace place) : place_(place) {}

const Place& IPUDeviceContext::GetPlace() const { return place_; }

void IPUDeviceContext::Wait() const {
  /*! \brief  Wait for all operations completion in the stream. */
}

IPUDeviceContext::~IPUDeviceContext() {}

#endif

}  // namespace platform
}  // namespace paddle
