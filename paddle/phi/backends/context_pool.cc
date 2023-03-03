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

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/context_pool_utils.h"

namespace phi {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
bool allow_tf32_cublas = true;
void SetAllowTF32Cublas(bool active) { allow_tf32_cublas = active; }
bool AllowTF32Cublas() { return allow_tf32_cublas; }
bool allow_tf32_cudnn = true;
void SetAllowTF32Cudnn(bool active) { allow_tf32_cudnn = active; }
bool AllowTF32Cudnn() { return allow_tf32_cudnn; }
#endif  // PADDLE_WITH_CUDA

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

size_t DeviceContextPool::Size() const {
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

inline void EmplaceNativeContext(
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
  }
}

void EmplaceDeviceContexts(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    const std::vector<phi::Place>& places,
    bool disable_setting_default_stream_for_allocator,
    int stream_priority,
    EmplaceExternalContextFunc emplace_external_context_func) {
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
    EmplaceNativeContext(place_to_device_context,
                         p,
                         disable_setting_default_stream_for_allocator,
                         stream_priority);
    if (emplace_external_context_func != nullptr) {
      (*emplace_external_context_func)(
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
                        /*stream_priority=*/0,
                        emplace_external_context_func_);
}

}  // namespace phi
