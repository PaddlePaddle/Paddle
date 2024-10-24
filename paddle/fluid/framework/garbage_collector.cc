// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/platform/cuda_device_guard.h"
#endif
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/phi/core/platform/device/device_wrapper.h"

COMMON_DECLARE_double(eager_delete_tensor_gb);
COMMON_DECLARE_double(memory_fraction_of_eager_deletion);
COMMON_DECLARE_bool(fast_eager_deletion_mode);

namespace paddle::framework {

GarbageCollector::GarbageCollector(const phi::Place &place,
                                   size_t max_memory_size)
    : garbages_(std::make_unique<GarbageQueue>()),
      mutex_(nullptr),
      max_memory_size_((std::max)(max_memory_size, static_cast<size_t>(1))) {
  dev_ctx_ = phi::DeviceContextPool::Instance().Get(place);
  if (max_memory_size_ > 1) {
    mutex_ = std::make_unique<std::mutex>();
  }
}

CPUGarbageCollector::CPUGarbageCollector(const phi::CPUPlace &place,
                                         size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {}

void CPUGarbageCollector::ClearCallback(const std::function<void()> &callback) {
  callback();
}

#ifdef PADDLE_WITH_XPU
XPUGarbageCollector::XPUGarbageCollector(const phi::XPUPlace &place,
                                         size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {}
void XPUGarbageCollector::ClearCallback(const std::function<void()> &callback) {
  callback();
}
#endif

#ifdef PADDLE_WITH_IPU
IPUGarbageCollector::IPUGarbageCollector(const phi::IPUPlace &place,
                                         size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {}
void IPUGarbageCollector::ClearCallback(const std::function<void()> &callback) {
  callback();
}
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
UnsafeFastGPUGarbageCollector::UnsafeFastGPUGarbageCollector(
    const phi::GPUPlace &place, size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {}

void UnsafeFastGPUGarbageCollector::ClearCallback(
    const std::function<void()> &callback) {
  callback();
}

DefaultStreamGarbageCollector::DefaultStreamGarbageCollector(
    const phi::GPUPlace &place, size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {}

void DefaultStreamGarbageCollector::Wait() const {
  static_cast<phi::GPUContext *>(this->dev_ctx_)->WaitStreamCallback();
}

void DefaultStreamGarbageCollector::ClearCallback(
    const std::function<void()> &callback) {
  static_cast<phi::GPUContext *>(this->dev_ctx_)->AddStreamCallback(callback);
}

StreamGarbageCollector::StreamGarbageCollector(const phi::GPUPlace &place,
                                               size_t max_memory_size)
    : GarbageCollector(place, max_memory_size),
      stream_(nullptr),
      callback_manager_(nullptr) {
  platform::CUDADeviceGuard guard(place.device);
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&stream_));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream_));
  callback_manager_ =
      std::make_unique<platform::StreamCallbackManager<gpuStream_t>>(stream_);
#endif
}

StreamGarbageCollector::~StreamGarbageCollector() {
  auto place = this->dev_ctx_->GetPlace();
  platform::CUDADeviceGuard guard(place.device);
  platform::GpuStreamSync(stream_);
  platform::GpuDestroyStream(stream_);
}

gpuStream_t StreamGarbageCollector::stream() const { return stream_; }

void StreamGarbageCollector::Wait() const { callback_manager_->Wait(); }

void StreamGarbageCollector::ClearCallback(
    const std::function<void()> &callback) {
  callback_manager_->AddCallback(callback);
}

CUDAPinnedGarbageCollector::CUDAPinnedGarbageCollector(
    const phi::GPUPinnedPlace &place, size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {}

void CUDAPinnedGarbageCollector::ClearCallback(
    const std::function<void()> &callback) {
  callback();
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
CustomDefaultStreamGarbageCollector::CustomDefaultStreamGarbageCollector(
    const phi::CustomPlace &place, size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {}

void CustomDefaultStreamGarbageCollector::Wait() const {
  static_cast<phi::CustomContext *>(this->dev_ctx_)->WaitStreamCallback();
}

void CustomDefaultStreamGarbageCollector::ClearCallback(
    const std::function<void()> &callback) {
  static_cast<phi::CustomContext *>(this->dev_ctx_)
      ->AddStreamCallback(callback);
}

CustomDeviceUnsafeFastGarbageCollector::CustomDeviceUnsafeFastGarbageCollector(
    const phi::CustomPlace &place, size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {}

void CustomDeviceUnsafeFastGarbageCollector::ClearCallback(
    const std::function<void()> &callback) {
  callback();
}

CustomStreamGarbageCollector::CustomStreamGarbageCollector(
    const phi::CustomPlace &place, size_t max_memory_size)
    : GarbageCollector(place, max_memory_size) {
  phi::DeviceGuard guard(place);
  stream_ = std::make_unique<phi::stream::Stream>();
  stream_->Init(place);
  callback_manager_ = std::make_unique<phi::CallbackManager>(stream_.get());
}

CustomStreamGarbageCollector::~CustomStreamGarbageCollector() {
  phi::DeviceGuard guard(this->dev_ctx_->GetPlace());
  stream_->Synchronize();
  stream_->Destroy();
}

phi::stream::Stream *CustomStreamGarbageCollector::stream() const {
  return stream_.get();
}

void CustomStreamGarbageCollector::Wait() const { callback_manager_->Wait(); }

void CustomStreamGarbageCollector::ClearCallback(
    const std::function<void()> &callback) {
  callback_manager_->AddCallback(callback);
}
#endif

int64_t GetEagerDeletionThreshold() {
  return FLAGS_eager_delete_tensor_gb < 0
             ? -1
             : static_cast<int64_t>(FLAGS_eager_delete_tensor_gb *
                                    (static_cast<int64_t>(1) << 30));
}

bool IsFastEagerDeletionModeEnabled() { return FLAGS_fast_eager_deletion_mode; }

void SetEagerDeletionMode(double threshold, double fraction, bool fast_mode) {
  FLAGS_eager_delete_tensor_gb = threshold;
  FLAGS_memory_fraction_of_eager_deletion = fraction;
  FLAGS_fast_eager_deletion_mode = fast_mode;
}

double GetEagerDeletionMemoryFraction() {
  return FLAGS_memory_fraction_of_eager_deletion;
}

std::unique_ptr<GarbageCollector> CreateGarbageCollector(
    const phi::Place &place, const size_t max_memory_size) {
  std::unique_ptr<GarbageCollector> gc = nullptr;
  if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (IsFastEagerDeletionModeEnabled()) {
      gc = std::make_unique<UnsafeFastGPUGarbageCollector>(place,
                                                           max_memory_size);
    } else {
      gc = std::make_unique<DefaultStreamGarbageCollector>(place,
                                                           max_memory_size);
    }
#else
    PADDLE_THROW(
        common::errors::Unimplemented("No GPU gc found in CPU/XPU paddle"));
#endif
  } else if (phi::is_cpu_place(place)) {
    gc = std::make_unique<CPUGarbageCollector>(place, max_memory_size);
  } else if (phi::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    gc = std::make_unique<XPUGarbageCollector>(place, max_memory_size);
#else
    PADDLE_THROW(
        common::errors::Unimplemented("No XPU gc found in CPU/GPU paddle"));
#endif
  } else if (phi::is_ipu_place(place)) {
#ifdef PADDLE_WITH_IPU
    gc = std::make_unique<IPUGarbageCollector>(place, max_memory_size);
#else
    PADDLE_THROW(
        common::errors::Unimplemented("No IPU gc found in CPU/IPU paddle"));
#endif
  } else if (phi::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    if (IsFastEagerDeletionModeEnabled()) {
      VLOG(4) << "Use unsafe fast gc for " << place << ".";
      gc = std::make_unique<CustomDeviceUnsafeFastGarbageCollector>(
          place, max_memory_size);
    } else {
      VLOG(4) << "Use default stream gc for " << place << ".";
      gc = std::make_unique<CustomDefaultStreamGarbageCollector>(
          place, max_memory_size);
    }
#else
    PADDLE_THROW(common::errors::Unimplemented("No CustomDevice gc found"));
#endif
  }
  return std::unique_ptr<GarbageCollector>(gc.release());
}

}  // namespace paddle::framework
