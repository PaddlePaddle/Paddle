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

#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/platform/device/mlu/device_context.h"
#include "paddle/fluid/platform/device/mlu/device_context_allocator.h"
#endif

#include "paddle/phi/backends/context_pool_utils.h"

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
  } else if (platform::is_npu_place(place)) {
    return platform::DeviceType::NPU;
  } else if (platform::is_mlu_place(place)) {
    return platform::DeviceType::MLU;
  } else if (platform::is_custom_place(place)) {
    return platform::DeviceType::CUSTOM_DEVICE;
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported place %s to convert into platform::DeviceType.", place));
  }
}

void EmplaceExternalContext(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    const platform::Place& place,
    bool disable_setting_default_stream_for_allocator,
    int stream_priority) {
  if (platform::is_cuda_pinned_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    phi::EmplaceDeviceContext<CUDAPinnedDeviceContext>(
        place_to_device_context,
        place,
        disable_setting_default_stream_for_allocator,
        /*unused*/ stream_priority);
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "CUDAPlace is not supported. Please re-compile with WITH_GPU "
        "option."));
#endif
  } else if (platform::is_mlu_place(place)) {
#ifdef PADDLE_WITH_MLU
    phi::EmplaceDeviceContext<MLUDeviceContext>(
        place_to_device_context,
        place,
        disable_setting_default_stream_for_allocator,
        /*unused*/ stream_priority);
#else
    PADDLE_THROW(
        platform::errors::Unimplemented("MLUPlace is not supported. Please "
                                        "re-compile with WITH_MLU option."));
#endif
  } else if (platform::is_ipu_place(place)) {
#ifdef PADDLE_WITH_IPU
    phi::EmplaceDeviceContext<IPUDeviceContext>(
        place_to_device_context,
        place,
        disable_setting_default_stream_for_allocator,
        /*unused*/ stream_priority);
#else
    PADDLE_THROW(
        platform::errors::Unimplemented("IPUPlace is not supported. Please "
                                        "re-compile with WITH_IPU option."));
#endif
  } else if (platform::is_npu_place(place)) {
#ifdef PADDLE_WITH_ASCEND_CL
    phi::EmplaceDeviceContext<NPUDeviceContext>(
        place_to_device_context,
        place,
        disable_setting_default_stream_for_allocator,
        /*unused*/ stream_priority);
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "NPUPlace is not supported. Please "
        "re-compile with WITH_ASCEND_CL option."));
#endif
  } else if (platform::is_npu_pinned_place(place)) {
#ifdef PADDLE_WITH_ASCEND_CL
    phi::EmplaceDeviceContext<NPUPinnedDeviceContext>(
        place_to_device_context,
        place,
        disable_setting_default_stream_for_allocator,
        /*unused*/ stream_priority);
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "NPUPinnedPlace is not supported. Please re-compile with "
        "WITH_ASCEND_CL "
        "option."));
#endif
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

#ifdef PADDLE_WITH_ASCEND_CL
NPUDeviceContext::NPUDeviceContext(NPUPlace place) : place_(place) {
  NPUDeviceGuard guard(place_.device);
  // PADDLE_ENFORCE_NPU_SUCCESS(aclrtCreateContext(&context_, place_.device));
  // NOTE(zhiqiu): Usually, no need to create context explicitly,
  // ACL creates a default context which contains 1 default stream
  // and 1 sync strean after aclrtSetDevice.
  platform::GetCurrentNPUContext(&context_);
  stream_.reset(new stream::NPUStream(place));
}

NPUDeviceContext::~NPUDeviceContext() {
  // NPUDeviceGuard guard(place_.device);
  // PADDLE_ENFORCE_NPU_SUCCESS(aclrtDestroyContext(context_));
}

void NPUDeviceContext::Wait() const {
  platform::RecordEvent record_event(
      "NPUDeviceContext/wait", platform::TracerEventType::UserDefined, 2);
  VLOG(4) << "NPU context(" << this << ")  Wait";
  stream_->Wait();
}

aclrtStream NPUDeviceContext::stream() const { return stream_->raw_stream(); }

const Place& NPUDeviceContext::GetPlace() const { return place_; }

aclrtContext NPUDeviceContext::context() const { return context_; }

NPUPinnedDeviceContext::NPUPinnedDeviceContext() {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

NPUPinnedDeviceContext::NPUPinnedDeviceContext(NPUPinnedPlace place)
    : place_(place) {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* NPUPinnedDeviceContext::eigen_device() const {
  return eigen_device_.get();
}

const Place& NPUPinnedDeviceContext::GetPlace() const { return place_; }

#endif

}  // namespace platform
}  // namespace paddle
