// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>

#include <utility>

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace c10 {

enum class EventFlag { PYTORCH_DEFAULT, BACKEND_DEFAULT, INVALID };

struct Event final {
 public:
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#ifdef PADDLE_WITH_HIP
  using BackendEvent = hipEvent_t;
  using BackendStream = hipStream_t;
#else
  using BackendEvent = cudaEvent_t;
  using BackendStream = cudaStream_t;
#endif
#endif

  Event() = delete;
  Event(const DeviceType device_type,
        const EventFlag flag = EventFlag::PYTORCH_DEFAULT)
      : device_type_(device_type), flag_(flag) {}

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  Event(Event&& other) noexcept { MoveFrom(std::move(other)); }
  Event& operator=(Event&& other) noexcept {
    if (this != &other) {
      DestroyBackendEvent();
      MoveFrom(std::move(other));
    }
    return *this;
  }

  ~Event() { DestroyBackendEvent(); }

  Device device() const noexcept { return Device(device_type_, device_index_); }
  DeviceType device_type() const noexcept { return device_type_; }
  DeviceIndex device_index() const noexcept { return device_index_; }
  EventFlag flag() const noexcept { return flag_; }
  bool was_marked_for_recording() const noexcept {
    return was_marked_for_recording_;
  }

  void recordOnce(const Stream& stream) {
    if (!was_marked_for_recording_) {
      record(stream);
    }
  }

  void record(const Stream& stream) {
    TORCH_CHECK(stream.device_type() == device_type_,
                "Event device type ",
                device_type_,
                " does not match recording stream's device type ",
                stream.device_type(),
                ".");
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (device_type_ == DeviceType::CUDA) {
      RecordBackendEvent(static_cast<BackendStream>(stream.native_handle()),
                         stream.device_index());
      return;
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support events.");
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void record(const c10::cuda::CUDAStream& stream) { record(stream.unwrap()); }
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // TODO(youge325): Remove after DeepEP paddle branch is updated to use
  // c10::Stream
#ifdef PADDLE_WITH_HIP
  void record(const hipStream_t& stream) {
    TORCH_CHECK(device_type_ == DeviceType::CUDA,
                "Raw hipStream_t recording is only supported for CUDA events.");
    RecordBackendEvent(stream, phi::backends::gpu::GetCurrentDeviceId());
  }
#else
  void record(const cudaStream_t& stream) {
    TORCH_CHECK(
        device_type_ == DeviceType::CUDA,
        "Raw cudaStream_t recording is only supported for CUDA events.");
    RecordBackendEvent(stream, phi::backends::gpu::GetCurrentDeviceId());
  }
#endif
#endif

  void block(const Stream& stream) const {
    if (!was_marked_for_recording_) {
      return;
    }
    TORCH_CHECK(stream.device_type() == device_type_,
                "Event device type ",
                device_type_,
                " does not match blocking stream's device type ",
                stream.device_type(),
                ".");
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (device_type_ == DeviceType::CUDA && backend_event_) {
      TORCH_CHECK(device_index_ == stream.device_index(),
                  "Event device index ",
                  static_cast<int>(device_index_),
                  " does not match blocking stream's device index ",
                  static_cast<int>(stream.device_index()),
                  ".");
      c10::cuda::CUDAGuard guard(device_index_);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipStreamWaitEvent(static_cast<BackendStream>(stream.native_handle()),
                             backend_event_,
                             0));
#else
      C10_CUDA_CHECK(cudaStreamWaitEvent(
          static_cast<BackendStream>(stream.native_handle()),
          backend_event_,
          0));
#endif
      return;
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support events.");
  }

  bool query() const {
    if (!was_marked_for_recording_) {
      return true;
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (device_type_ == DeviceType::CUDA && backend_event_) {
#ifdef PADDLE_WITH_HIP
      const auto err = hipEventQuery(backend_event_);
      if (err == hipSuccess) {
        return true;
      }
      if (err != hipErrorNotReady) {
        PADDLE_ENFORCE_GPU_SUCCESS(err);
      } else {
        (void)hipGetLastError();
      }
#else
      const auto err = cudaEventQuery(backend_event_);
      if (err == cudaSuccess) {
        return true;
      }
      if (err != cudaErrorNotReady) {
        C10_CUDA_CHECK(err);
      } else {
        (void)cudaGetLastError();
      }
#endif
      return false;
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support events.");
    return true;
  }

  double elapsedTime(const Event& event) const {
    TORCH_CHECK(event.device_type() == device_type_,
                "Event device type ",
                device_type_,
                " does not match other's device type ",
                event.device_type(),
                ".");
    TORCH_CHECK(
        flag_ == EventFlag::BACKEND_DEFAULT &&
            event.flag_ == EventFlag::BACKEND_DEFAULT,
        "Both events must be created with argument 'enable_timing=True'.");
    TORCH_CHECK(
        was_marked_for_recording_ && event.was_marked_for_recording_,
        "Both events must be recorded before calculating elapsed time.");
    TORCH_CHECK(
        query() && event.query(),
        "Both events must be completed before calculating elapsed time.");
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (device_type_ == DeviceType::CUDA && backend_event_ &&
        event.backend_event_) {
      TORCH_CHECK(device_index_ == event.device_index_,
                  "Event device index ",
                  static_cast<int>(device_index_),
                  " does not match other's device index ",
                  static_cast<int>(event.device_index_),
                  ".");
      c10::cuda::CUDAGuard guard(device_index_);
      float time_ms = 0.0f;
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipEventElapsedTime(&time_ms, backend_event_, event.backend_event_));
#else
      C10_CUDA_CHECK(
          cudaEventElapsedTime(&time_ms, backend_event_, event.backend_event_));
#endif
      return static_cast<double>(time_ms);
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support event elapsedTime.");
    return 0.0;
  }

  void* eventId() const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    return backend_event_;
#else
    return nullptr;
#endif
  }

  void synchronize() const {
    if (!was_marked_for_recording_) {
      return;
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (device_type_ == DeviceType::CUDA && backend_event_) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventSynchronize(backend_event_));
#else
      C10_CUDA_CHECK(cudaEventSynchronize(backend_event_));
#endif
      return;
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support events.");
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#ifdef PADDLE_WITH_HIP
  hipEvent_t cuda_event() const { return backend_event_; }
#else
  cudaEvent_t cuda_event() const { return backend_event_; }
#endif
#endif

 private:
  DeviceType device_type_;
  DeviceIndex device_index_ = -1;
  EventFlag flag_ = EventFlag::PYTORCH_DEFAULT;
  bool was_marked_for_recording_ = false;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  BackendEvent backend_event_ = nullptr;

  static unsigned int BackendEventCreateFlags(EventFlag flag) {
    switch (flag) {
      case EventFlag::PYTORCH_DEFAULT:
#ifdef PADDLE_WITH_HIP
        return hipEventDisableTiming;
#else
        return cudaEventDisableTiming;
#endif
      case EventFlag::BACKEND_DEFAULT:
#ifdef PADDLE_WITH_HIP
        return hipEventDefault;
#else
        return cudaEventDefault;
#endif
      default:
        TORCH_CHECK(false, "CUDA event received unknown flag");
    }
  }

  void EnsureBackendEventCreated(DeviceIndex stream_device_index) {
    if (backend_event_) {
      return;
    }
    c10::cuda::CUDAGuard guard(stream_device_index);
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventCreateWithFlags(
        &backend_event_, BackendEventCreateFlags(flag_)));
#else
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&backend_event_,
                                            BackendEventCreateFlags(flag_)));
#endif
  }

  void RecordBackendEvent(BackendStream stream,
                          DeviceIndex stream_device_index) {
    TORCH_CHECK(device_index_ == -1 || device_index_ == stream_device_index,
                "Event device index ",
                static_cast<int>(device_index_),
                " does not match recording stream's device index ",
                static_cast<int>(stream_device_index),
                ".");
    EnsureBackendEventCreated(stream_device_index);
    c10::cuda::CUDAGuard guard(stream_device_index);
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(backend_event_, stream));
#else
    C10_CUDA_CHECK(cudaEventRecord(backend_event_, stream));
#endif
    device_index_ = stream_device_index;
    was_marked_for_recording_ = true;
  }

  void DestroyBackendEvent() noexcept {
    if (!backend_event_) {
      return;
    }
    try {
      c10::cuda::CUDAGuard guard(device_index_);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(backend_event_));
#else
      C10_CUDA_CHECK(cudaEventDestroy(backend_event_));
#endif
    } catch (...) {
    }
    backend_event_ = nullptr;
  }
#else
  void DestroyBackendEvent() noexcept {}
#endif

  void MoveFrom(Event&& other) noexcept {
    device_type_ = other.device_type_;
    device_index_ = other.device_index_;
    flag_ = other.flag_;
    was_marked_for_recording_ = other.was_marked_for_recording_;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    backend_event_ = std::exchange(other.backend_event_, nullptr);
#endif
    other.device_index_ = -1;
    other.was_marked_for_recording_ = false;
  }
};

}  // namespace c10

namespace torch {
using c10::Event;
}  // namespace torch
