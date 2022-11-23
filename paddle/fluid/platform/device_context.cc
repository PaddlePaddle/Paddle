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
#include "paddle/fluid/framework/expect.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/allocator.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/memory/allocation/cuda_device_context_allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif

#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/platform/device/mlu/device_context.h"
#include "paddle/fluid/platform/device/mlu/device_context_allocator.h"
#endif

namespace paddle {
namespace platform {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
bool allow_tf32_cublas = true;
void SetAllowTF32Cublas(bool active) { allow_tf32_cublas = active; }
bool AllowTF32Cublas() { return allow_tf32_cublas; }

bool allow_tf32_cudnn = true;
void SetAllowTF32Cudnn(bool active) { allow_tf32_cudnn = active; }
bool AllowTF32Cudnn() { return allow_tf32_cudnn; }
#endif  // PADDLE_WITH_CUDA

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

static DeviceContextPool* pool = nullptr;

DeviceContextPool& DeviceContextPool::Instance() {
  PADDLE_ENFORCE_NOT_NULL(pool,
                          phi::errors::PreconditionNotMet(
                              "Need to Create DeviceContextPool firstly!"));
  return *pool;
}

/*! \brief  Create should only called by Init function */
DeviceContextPool& DeviceContextPool::Init(
    const std::vector<platform::Place>& places) {
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

platform::DeviceContext* DeviceContextPool::Get(const platform::Place& place) {
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
    PADDLE_THROW(platform::errors::Unimplemented(
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

template <typename DevCtx>
std::unique_ptr<DeviceContext> CreateDeviceContext(
    const platform::Place& p,
    bool disable_setting_default_stream_for_allocator = false) {
  using PtrType = std::unique_ptr<DeviceContext>;
  auto* dev_ctx = new DevCtx(p);
  if (is_gpu_place(p)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto* cuda_ctx = dynamic_cast<phi::GPUContext*>(dev_ctx);
    PADDLE_ENFORCE_NOT_NULL(
        cuda_ctx,
        platform::errors::InvalidArgument(
            "Failed to dynamic_cast dev_ctx into phi::GPUContext."));

    auto& instance = memory::allocation::AllocatorFacade::Instance();
    if (!disable_setting_default_stream_for_allocator) {
      instance.SetDefaultStream(CUDAPlace(p.GetDeviceId()), cuda_ctx->stream());
    }
    dev_ctx->SetAllocator(instance.GetAllocator(p).get());
    dev_ctx->SetPinnedAllocator(
        instance.GetAllocator(paddle::platform::CUDAPinnedPlace()).get());

    cuda_ctx->PartialInitWithAllocator();
    dev_ctx->SetGenerator(
        framework::DefaultCUDAGenerator(p.GetDeviceId()).get());
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
  return PtrType(dev_ctx);
}

template <typename DevCtx>
inline void EmplaceDeviceContext(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    platform::Place place,
    bool disable_setting_default_stream_for_allocator) {
  // lazy evaluation. i.e., only create device context at first `Get`
  place_to_device_context->emplace(
      place,
      std::async(std::launch::deferred,
                 CreateDeviceContext<DevCtx>,
                 place,
                 disable_setting_default_stream_for_allocator));
}

void EmplaceDeviceContexts(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    const std::vector<platform::Place>& places,
    bool disable_setting_default_stream_for_allocator) {
  PADDLE_ENFORCE_GT(
      places.size(),
      0,
      platform::errors::InvalidArgument("The number of platform places should "
                                        "be larger than 0. But received %d.",
                                        places.size()));

  std::set<Place> set;
  for (auto& p : places) {
    set.insert(p);
  }

  for (auto& p : set) {
    if (platform::is_cpu_place(p)) {
#ifdef PADDLE_WITH_MKLDNN
      EmplaceDeviceContext<MKLDNNDeviceContext>(
          place_to_device_context,
          p,
          disable_setting_default_stream_for_allocator);
#else
      EmplaceDeviceContext<phi::CPUContext>(
          place_to_device_context,
          p,
          disable_setting_default_stream_for_allocator);
#endif
    } else if (platform::is_gpu_place(p)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      EmplaceDeviceContext<phi::GPUContext>(
          place_to_device_context,
          p,
          disable_setting_default_stream_for_allocator);
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("CUDAPlace is not supported. Please "
                                          "re-compile with WITH_GPU option."));
#endif
    } else if (platform::is_cuda_pinned_place(p)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      EmplaceDeviceContext<CUDAPinnedDeviceContext>(
          place_to_device_context,
          p,
          disable_setting_default_stream_for_allocator);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "CUDAPlace is not supported. Please re-compile with WITH_GPU "
          "option."));
#endif
    } else if (platform::is_xpu_place(p)) {
#ifdef PADDLE_WITH_XPU
      EmplaceDeviceContext<XPUDeviceContext>(
          place_to_device_context,
          p,
          disable_setting_default_stream_for_allocator);
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("XPUPlace is not supported. Please "
                                          "re-compile with WITH_XPU option."));
#endif
    } else if (platform::is_mlu_place(p)) {
#ifdef PADDLE_WITH_MLU
      EmplaceDeviceContext<MLUDeviceContext>(
          place_to_device_context,
          p,
          disable_setting_default_stream_for_allocator);
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("MLUPlace is not supported. Please "
                                          "re-compile with WITH_MLU option."));
#endif
    } else if (platform::is_ipu_place(p)) {
#ifdef PADDLE_WITH_IPU
      EmplaceDeviceContext<IPUDeviceContext>(
          place_to_device_context,
          p,
          disable_setting_default_stream_for_allocator);
#else
      PADDLE_THROW(
          platform::errors::Unimplemented("IPUPlace is not supported. Please "
                                          "re-compile with WITH_IPU option."));
#endif
    } else if (platform::is_npu_place(p)) {
#ifdef PADDLE_WITH_ASCEND_CL
      EmplaceDeviceContext<NPUDeviceContext>(
          place_to_device_context,
          p,
          disable_setting_default_stream_for_allocator);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "NPUPlace is not supported. Please "
          "re-compile with WITH_ASCEND_CL option."));
#endif
    } else if (platform::is_npu_pinned_place(p)) {
#ifdef PADDLE_WITH_ASCEND_CL
      EmplaceDeviceContext<NPUPinnedDeviceContext>(
          place_to_device_context,
          p,
          disable_setting_default_stream_for_allocator);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "NPUPinnedPlace is not supported. Please re-compile with "
          "WITH_ASCEND_CL "
          "option."));
#endif
    } else if (platform::is_custom_place(p)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      EmplaceDeviceContext<CustomDeviceContext>(
          place_to_device_context,
          p,
          disable_setting_default_stream_for_allocator);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "CustomPlace is not supported. Please re-compile with "
          "WITH_CUSTOM_DEVICE "
          "option."));
#endif
    }
  }
}

DeviceContextPool::DeviceContextPool(
    const std::vector<platform::Place>& places) {
  EmplaceDeviceContexts(&device_contexts_,
                        places,
                        /*disable_setting_default_stream_for_allocator=*/false);
}

#ifdef PADDLE_WITH_IPU
IPUDeviceContext::IPUDeviceContext(IPUPlace place) : place_(place) {}

const Place& IPUDeviceContext::GetPlace() const { return place_; }

void IPUDeviceContext::Wait() const {
  /*! \brief  Wait for all operations completion in the stream. */
}

IPUDeviceContext::~IPUDeviceContext() {}

#endif
#ifdef PADDLE_WITH_XPU
XPUDeviceContext::XPUDeviceContext() : phi::XPUContext() {
  phi::XPUContext::Init();
}

XPUDeviceContext::~XPUDeviceContext() {}

XPUDeviceContext::XPUDeviceContext(XPUPlace place) : phi::XPUContext(place) {
  phi::XPUContext::Init();
  LOG_FIRST_N(WARNING, 1) << "Please NOTE: xpu device: "
                          << static_cast<int>(place.device);
}
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

CUDAPinnedDeviceContext::CUDAPinnedDeviceContext() {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

CUDAPinnedDeviceContext::CUDAPinnedDeviceContext(CUDAPinnedPlace place)
    : place_(place) {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* CUDAPinnedDeviceContext::eigen_device() const {
  return eigen_device_.get();
}

const Place& CUDAPinnedDeviceContext::GetPlace() const { return place_; }
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
CustomDeviceContext::CustomDeviceContext(CustomPlace place)
    : phi::CustomContext(place) {
  Init();
  stream_.reset(new phi::stream::Stream(place, stream()));
}

CustomDeviceContext::~CustomDeviceContext() {}
#endif
}  // namespace platform
}  // namespace paddle
