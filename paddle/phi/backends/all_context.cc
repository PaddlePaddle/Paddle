/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/all_context.h"

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

static DeviceContextPool* pool = nullptr;

DeviceContextPool& DeviceContextPool::Instance() {
  PADDLE_ENFORCE_NOT_NULL(pool,
                          phi::errors::PreconditionNotMet(
                              "Need to Create DeviceContextPool firstly!"));
  return *pool;
}

EmplaceExternalContextFunc DeviceContextPool::emplace_external_context_func_ =
    nullptr;

/*! \brief  Create should only called by Init function */
DeviceContextPool& DeviceContextPool::Init(
    const std::vector<phi::Place>& places, EmplaceExternalContextFunc func) {
  emplace_external_context_func_ = func;
  if (pool == nullptr) {
    pool = new DeviceContextPool(places);
  }
  return *pool;
}

bool DeviceContextPool::IsInitialized() { return pool != nullptr; }

void DeviceContextPool::SetPool(DeviceContextPool* dev_pool) {
  pool = dev_pool;
}

thread_local const std::map<Place,
                            std::shared_future<std::unique_ptr<DeviceContext>>>*
    DeviceContextPool::external_device_contexts_ = nullptr;

phi::DeviceContext* DeviceContextPool::Get(const phi::Place& place) {
  VLOG(6) << "DeviceContextPool Get: " << place;
  const std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
      ptr;
  if (external_device_contexts_ && external_device_contexts_->count(place)) {
    ptr = external_device_contexts_;
  } else {
    ptr = &device_contexts_;
  }

  auto it = ptr->find(place);
  if (it == ptr->end()) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Place %s is not supported. Please check that your paddle compiles "
        "with WITH_GPU, WITH_XPU, WITH_IPU, WITH_MLU or WITH_ASCEND_CL option "
        "or check "
        "that your train process set the correct device id if you use "
        "Executor.",
        place));
  }
  return it->second.get().get();
}

size_t DeviceContextPool::size() const {
  if (external_device_contexts_) {
    return external_device_contexts_->size();
  }
  return device_contexts_.size();
}

const std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>&
DeviceContextPool::device_contexts() const {
  if (external_device_contexts_) {
    return *external_device_contexts_;
  }
  return device_contexts_;
}

void DeviceContextPool::SetDeviceContexts(
    const std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        dev_ctxs) {
  external_device_contexts_ = dev_ctxs;
}

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
std::unique_ptr<DeviceContext> CreateDeviceContext(
    const phi::Place& p,
    bool disable_setting_default_stream_for_allocator = false,
    int stream_priority = 0) {
  using PtrType = std::unique_ptr<DeviceContext>;

  DevCtx* dev_ctx = ConstructDevCtx<DevCtx>(p, stream_priority);

  if (is_gpu_place(p)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto* cuda_ctx = dynamic_cast<phi::GPUContext*>(dev_ctx);
    PADDLE_ENFORCE_NOT_NULL(
        cuda_ctx,
        phi::errors::InvalidArgument(
            "Failed to dynamic_cast dev_ctx into phi::GPUContext."));

    auto& instance = memory::allocation::AllocatorFacade::Instance();
    if (!disable_setting_default_stream_for_allocator) {
      instance.SetDefaultStream(CUDAPlace(p.GetDeviceId()), cuda_ctx->stream());
    }
    dev_ctx->SetAllocator(instance.GetAllocator(p, cuda_ctx->stream()).get());
    dev_ctx->SetPinnedAllocator(
        instance.GetAllocator(phi::GPUPinnedPlace()).get());

    cuda_ctx->PartialInitWithAllocator();
    dev_ctx->SetGenerator(
        framework::DefaultCUDAGenerator(p.GetDeviceId()).get());
#endif
  } else if (is_xpu_place(p)) {
#if defined(PADDLE_WITH_XPU)
    dev_ctx->SetAllocator(
        memory::allocation::AllocatorFacade::Instance().GetAllocator(p).get());
    dev_ctx->SetGenerator(
        framework::DefaultXPUGenerator(p.GetDeviceId()).get());
#endif
  } else {
    dev_ctx->SetAllocator(
        memory::allocation::AllocatorFacade::Instance().GetAllocator(p).get());
    dev_ctx->SetGenerator(framework::DefaultCPUGenerator().get());
  }
  dev_ctx->SetHostGenerator(framework::DefaultCPUGenerator().get());
  dev_ctx->SetHostAllocator(memory::allocation::AllocatorFacade::Instance()
                                .GetAllocator(platform::CPUPlace())
                                .get());
  dev_ctx->SetZeroAllocator(memory::allocation::AllocatorFacade::Instance()
                                .GetZeroAllocator(p)
                                .get());
  dev_ctx->SetHostZeroAllocator(memory::allocation::AllocatorFacade::Instance()
                                    .GetZeroAllocator(platform::CPUPlace())
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

inline void EmplacePhiContext(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    const phi::Place& place,
    bool disable_setting_default_stream_for_allocator,
    int stream_priority) {
  if (place.GetType() == phi::AllocationType::CPU) {
#ifdef PADDLE_WITH_MKLDNN
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
        phi::errors::Unimplemented("CUDAPlace is not supported. Please "
                                   "re-compile with WITH_GPU option."));
#endif
  } else if (place.GetType() == phi::AllocationType::XPU) {
#ifdef PADDLE_WITH_XPU
    EmplaceDeviceContext<XPUDeviceContext>(
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
    EmplaceDeviceContext<CustomDeviceContext>(
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
  }
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

  for (auto& p : set) {
    EmplacePhiContext(place_to_device_context,
                      p,
                      disable_setting_default_stream_for_allocator,
                      stream_priority);
    if (emplace_external_context_func_ != nullptr) {
      (*emplace_external_context_func_)(
          place_to_device_context,
          p,
          disable_setting_default_stream_for_allocator,
          stream_priority);
    }
  }
}

DeviceContextPool::DeviceContextPool(const std::vector<phi::Place>& places) {
  EmplaceDeviceContexts(&device_contexts_,
                        places,
                        /*disable_setting_default_stream_for_allocator=*/false,
                        /*stream_priority=*/0);
}

}  // namespace phi
