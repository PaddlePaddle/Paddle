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
#include <c10/cuda/CUDAStream.h>
#include <queue>
namespace c10 {

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

struct Event final {
 public:
  Event(const DeviceType &type) {
    // device_type is useless, only for compatibility
    cuda_event_ = EventPool::Instance().CreateCudaEventFromPool();
  }

  void record(const Stream &stream) {
    C10_CUDA_CHECK(cudaEventRecord(
        cuda_event_, static_cast<cudaStream_t>(stream.native_handle())));
  }

  void record(const c10::cuda::CUDAStream &stream) { record(stream.unwrap()); }

  void block(const Stream &stream) const {
    C10_CUDA_CHECK(cudaStreamWaitEvent(
        static_cast<cudaStream_t>(stream.native_handle()), cuda_event_, 0));
  }

  cudaEvent_t cuda_event() const { return cuda_event_; }

 private:
  cudaEvent_t cuda_event_;
};

}  // namespace c10

namespace torch {
using c10::Event;
}  // namespace torch

#endif
