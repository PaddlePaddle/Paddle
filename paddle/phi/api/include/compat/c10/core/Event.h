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

#ifdef PADDLE_WITH_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace c10 {

enum class EventFlag { PYTORCH_DEFAULT, BACKEND_DEFAULT, INVALID };

struct Event final {
 public:
  Event() = delete;
  Event(const DeviceType device_type,
        const EventFlag flag = EventFlag::PYTORCH_DEFAULT)
      : device_type_(device_type), flag_(flag) {}

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  Event(Event&& other) noexcept { MoveFrom(std::move(other)); }
  Event& operator=(Event&& other) noexcept {
    if (this != &other) {
      DestroyCudaEvent();
      MoveFrom(std::move(other));
    }
    return *this;
  }

  ~Event() { DestroyCudaEvent(); }

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
#ifdef PADDLE_WITH_CUDA
    if (device_type_ == DeviceType::CUDA) {
      RecordCudaEvent(static_cast<cudaStream_t>(stream.native_handle()),
                      stream.device_index());
      return;
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support events.");
  }

#ifdef PADDLE_WITH_CUDA
  void record(const c10::cuda::CUDAStream& stream) { record(stream.unwrap()); }

  // TODO(youge325): Remove after DeepEP paddle branch is updated to use
  // c10::Stream
  void record(const cudaStream_t& stream) {
    TORCH_CHECK(
        device_type_ == DeviceType::CUDA,
        "Raw cudaStream_t recording is only supported for CUDA events.");
    RecordCudaEvent(stream, phi::backends::gpu::GetCurrentDeviceId());
  }
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
#ifdef PADDLE_WITH_CUDA
    if (device_type_ == DeviceType::CUDA && cuda_event_) {
      TORCH_CHECK(device_index_ == stream.device_index(),
                  "Event device index ",
                  static_cast<int>(device_index_),
                  " does not match blocking stream's device index ",
                  static_cast<int>(stream.device_index()),
                  ".");
      c10::cuda::CUDAGuard guard(device_index_);
      C10_CUDA_CHECK(cudaStreamWaitEvent(
          static_cast<cudaStream_t>(stream.native_handle()), cuda_event_, 0));
      return;
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support events.");
  }

  bool query() const {
    if (!was_marked_for_recording_) {
      return true;
    }
#ifdef PADDLE_WITH_CUDA
    if (device_type_ == DeviceType::CUDA && cuda_event_) {
      const auto err = cudaEventQuery(cuda_event_);
      if (err == cudaSuccess) {
        return true;
      }
      if (err != cudaErrorNotReady) {
        C10_CUDA_CHECK(err);
      } else {
        (void)cudaGetLastError();
      }
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
#ifdef PADDLE_WITH_CUDA
    if (device_type_ == DeviceType::CUDA && cuda_event_ && event.cuda_event_) {
      TORCH_CHECK(device_index_ == event.device_index_,
                  "Event device index ",
                  static_cast<int>(device_index_),
                  " does not match other's device index ",
                  static_cast<int>(event.device_index_),
                  ".");
      c10::cuda::CUDAGuard guard(device_index_);
      float time_ms = 0.0f;
      C10_CUDA_CHECK(
          cudaEventElapsedTime(&time_ms, cuda_event_, event.cuda_event_));
      return static_cast<double>(time_ms);
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support event elapsedTime.");
    return 0.0;
  }

  void* eventId() const {
#ifdef PADDLE_WITH_CUDA
    return cuda_event_;
#else
    return nullptr;
#endif
  }

  void synchronize() const {
    if (!was_marked_for_recording_) {
      return;
    }
#ifdef PADDLE_WITH_CUDA
    if (device_type_ == DeviceType::CUDA && cuda_event_) {
      C10_CUDA_CHECK(cudaEventSynchronize(cuda_event_));
      return;
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support events.");
  }

#ifdef PADDLE_WITH_CUDA
  cudaEvent_t cuda_event() const { return cuda_event_; }
#endif

 private:
  DeviceType device_type_;
  DeviceIndex device_index_ = -1;
  EventFlag flag_ = EventFlag::PYTORCH_DEFAULT;
  bool was_marked_for_recording_ = false;
#ifdef PADDLE_WITH_CUDA
  cudaEvent_t cuda_event_ = nullptr;

  static unsigned int CudaEventCreateFlags(EventFlag flag) {
    switch (flag) {
      case EventFlag::PYTORCH_DEFAULT:
        return cudaEventDisableTiming;
      case EventFlag::BACKEND_DEFAULT:
        return cudaEventDefault;
      default:
        TORCH_CHECK(false, "CUDA event received unknown flag");
    }
  }

  void EnsureCudaEventCreated(DeviceIndex stream_device_index) {
    if (cuda_event_) {
      return;
    }
    c10::cuda::CUDAGuard guard(stream_device_index);
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(&cuda_event_, CudaEventCreateFlags(flag_)));
  }

  void RecordCudaEvent(cudaStream_t stream, DeviceIndex stream_device_index) {
    TORCH_CHECK(device_index_ == -1 || device_index_ == stream_device_index,
                "Event device index ",
                static_cast<int>(device_index_),
                " does not match recording stream's device index ",
                static_cast<int>(stream_device_index),
                ".");
    EnsureCudaEventCreated(stream_device_index);
    c10::cuda::CUDAGuard guard(stream_device_index);
    C10_CUDA_CHECK(cudaEventRecord(cuda_event_, stream));
    device_index_ = stream_device_index;
    was_marked_for_recording_ = true;
  }

  void DestroyCudaEvent() noexcept {
    if (!cuda_event_) {
      return;
    }
    try {
      c10::cuda::CUDAGuard guard(device_index_);
      C10_CUDA_CHECK(cudaEventDestroy(cuda_event_));
    } catch (...) {
    }
    cuda_event_ = nullptr;
  }
#else
  void DestroyCudaEvent() noexcept {}
#endif

  void MoveFrom(Event&& other) noexcept {
    device_type_ = other.device_type_;
    device_index_ = other.device_index_;
    flag_ = other.flag_;
    was_marked_for_recording_ = other.was_marked_for_recording_;
#ifdef PADDLE_WITH_CUDA
    cuda_event_ = std::exchange(other.cuda_event_, nullptr);
#endif
    other.device_index_ = -1;
    other.was_marked_for_recording_ = false;
  }
};

}  // namespace c10

namespace torch {
using c10::Event;
}  // namespace torch
