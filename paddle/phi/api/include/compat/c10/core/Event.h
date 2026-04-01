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

#ifdef PADDLE_WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#include <queue>
#endif

namespace c10 {

enum class EventFlag { PYTORCH_DEFAULT, BACKEND_DEFAULT, INVALID };

#ifdef PADDLE_WITH_CUDA

class EventPool {
 public:
  EventPool();
  EventPool(const EventPool &) = delete;
  EventPool(EventPool &&) = delete;
  ~EventPool();

  cudaEvent_t CreateCudaEventFromPool();

  static EventPool &Instance();

 private:
  std::queue<cudaEvent_t> incomplished_events_;
  std::mutex mtx_;
};

EventPool &EventPool::Instance() {
  static EventPool pool;
  return pool;
}

EventPool::EventPool() {
  for (size_t i = 0; i < 1000; ++i) {
    cudaEvent_t new_event;
    C10_CUDA_CHECK(cudaEventCreate(&new_event));

    cudaEventRecord(new_event, 0);
    incomplished_events_.push(new_event);
  }
}

EventPool::~EventPool() {
  std::unique_lock<std::mutex> lock(mtx_);
  while (!incomplished_events_.empty()) {
    cudaEvent_t event = incomplished_events_.front();
    incomplished_events_.pop();
    if (cudaEventQuery(event) == cudaSuccess) {
      C10_CUDA_CHECK(cudaEventDestroy(event));
    }
  }
}

cudaEvent_t EventPool::CreateCudaEventFromPool() {
  std::unique_lock<std::mutex> lock(mtx_);

  const auto &CreateNewEvent = [&]() -> cudaEvent_t {
    cudaEvent_t new_event;
    C10_CUDA_CHECK(cudaEventCreate(&new_event));
    incomplished_events_.push(new_event);
    return new_event;
  };

  const auto &CreateNewOrReuseEvent = [&]() -> cudaEvent_t {
    cudaEvent_t front_event = incomplished_events_.front();
    incomplished_events_.pop();
    incomplished_events_.push(front_event);
    if (cudaEventQuery(front_event) == cudaSuccess) {
      return front_event;
    }
    return CreateNewEvent();
  };

  if (incomplished_events_.empty()) {
    return CreateNewEvent();
  }
  return CreateNewOrReuseEvent();
}

#endif  // PADDLE_WITH_CUDA

struct Event final {
 public:
  Event() = delete;
  Event(const DeviceType device_type,
        const EventFlag flag = EventFlag::PYTORCH_DEFAULT)
      : device_type_(device_type), flag_(flag) {
#ifdef PADDLE_WITH_CUDA
    if (device_type == DeviceType::CUDA) {
      cuda_event_ = EventPool::Instance().CreateCudaEventFromPool();
    }
#endif
  }

  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  Event(Event &&) = default;
  Event &operator=(Event &&) = default;
  ~Event() = default;

  Device device() const noexcept { return Device(device_type_, device_index_); }
  DeviceType device_type() const noexcept { return device_type_; }
  DeviceIndex device_index() const noexcept { return device_index_; }
  EventFlag flag() const noexcept { return flag_; }
  bool was_marked_for_recording() const noexcept {
    return was_marked_for_recording_;
  }

  void recordOnce(const Stream &stream) {
    if (!was_marked_for_recording_) record(stream);
  }

  void record(const Stream &stream) {
    TORCH_CHECK(
        stream.device_type() == device_type_,
        "Event device type does not match recording stream's device type.");
#ifdef PADDLE_WITH_CUDA
    if (device_type_ == DeviceType::CUDA && cuda_event_) {
      C10_CUDA_CHECK(cudaEventRecord(
          cuda_event_, static_cast<cudaStream_t>(stream.native_handle())));
      was_marked_for_recording_ = true;
      device_index_ = stream.device_index();
      return;
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support events.");
  }

#ifdef PADDLE_WITH_CUDA
  void record(const c10::cuda::CUDAStream &stream) { record(stream.unwrap()); }
#endif

  void block(const Stream &stream) const {
    if (!was_marked_for_recording_) return;
    TORCH_CHECK(
        stream.device_type() == device_type_,
        "Event device type does not match blocking stream's device type.");
#ifdef PADDLE_WITH_CUDA
    if (device_type_ == DeviceType::CUDA && cuda_event_) {
      C10_CUDA_CHECK(cudaStreamWaitEvent(
          static_cast<cudaStream_t>(stream.native_handle()), cuda_event_, 0));
      return;
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support events.");
  }

  bool query() const {
    if (!was_marked_for_recording_) return true;
#ifdef PADDLE_WITH_CUDA
    if (device_type_ == DeviceType::CUDA && cuda_event_) {
      return cudaEventQuery(cuda_event_) == cudaSuccess;
    }
#endif
    TORCH_CHECK(false, "Backend doesn't support events.");
    return true;
  }

  double elapsedTime(const Event &event) const {
    (void)event;
    return 0.0;
  }

  void *eventId() const {
#ifdef PADDLE_WITH_CUDA
    return cuda_event_;
#else
    return nullptr;
#endif
  }

  void synchronize() const {
#ifdef PADDLE_WITH_CUDA
    if (device_type_ == DeviceType::CUDA && cuda_event_) {
      C10_CUDA_CHECK(cudaEventSynchronize(cuda_event_));
    }
#endif
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
#endif
};

}  // namespace c10

namespace torch {
using c10::Event;
}  // namespace torch
