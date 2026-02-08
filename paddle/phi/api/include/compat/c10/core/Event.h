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

/**
 * A backend-generic movable, not copyable, not thread-safe event.
 *
 * The design of this event follows that of CUDA and HIP events. These events
 * are recorded and waited on by streams and can be rerecorded to,
 * each rerecording essentially creating a new version of the event.
 * For example, if (in CPU time), stream X is asked to record E,
 * stream Y waits on E, and stream X is asked to record E again, then Y will
 * wait for X to finish the first call to record and not the second, because
 * it's waiting on the first version of event E, not the second.
 * Querying an event only returns the status of its most recent version.
 *
 * Backend-generic events are implemented by this class and
 * impl::InlineEvent. In addition to these events there are also
 * some backend-specific events, like ATen's CUDAEvent. Each of these
 * classes has its own use.
 *
 * impl::InlineEvent<...> or a backend-specific event should be
 * preferred when the backend is known at compile time and known to
 * be compiled. Backend-specific events may have additional functionality.
 *
 * This Event should be used if a particular backend may not be available,
 * or the backend required is not known at compile time.
 *
 * These generic events are built on top of DeviceGuardImpls, analogous
 * to DeviceGuard and InlineDeviceGuard. The name "DeviceGuardImpls,"
 * is no longer entirely accurate, as these classes implement the
 * backend-specific logic for a generic backend interface.
 *
 * See DeviceGuardImplInterface.h for a list of all supported flags.
 */

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
  const auto &DestroyEvent = [](cudaEvent_t event) {
    C10_CUDA_CHECK(cudaEventDestroy(event));
  };
  const auto &CheckComplishAndDestroy = [&](cudaEvent_t event) -> bool {
    if (cudaEventQuery(event) == cudaSuccess) {
      DestroyEvent(event);
      return true;
    }
    if (cudaEventQuery(event) == cudaErrorNotReady) {
      // LOG(ERROR) << "event is not completed or when destroying event pool.";
      return false;
    }
    // LOG(ERROR) << "failed on cudaEventQuery when destroying event pool.";
    return false;
  };
  std::unique_lock<std::mutex> lock(mtx_);
  while (!incomplished_events_.empty()) {
    cudaEvent_t event = incomplished_events_.front();
    if (!CheckComplishAndDestroy(event)) {
      // LOG(ERROR) << "failed on destroying event when destroying event pool.";
    }
    incomplished_events_.pop();
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
  void record(const cudaStream_t &stream) {
    C10_CUDA_CHECK(cudaEventRecord(cuda_event_, stream));
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
