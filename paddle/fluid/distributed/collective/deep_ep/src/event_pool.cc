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

#include "paddle/fluid/distributed/collective/deep_ep/include/event_pool.h"
#include "glog/logging.h"

namespace deep_ep::detail {

EventPool &EventPool::Instance() {
  static EventPool pool;
  return pool;
}

EventPool::~EventPool() {
  const auto &DestroyEvent = [](cudaEvent_t event) {
    cudaError_t e = cudaEventDestroy(event);
    if (e != cudaSuccess) {
      LOG(ERROR) << "CUDA event destroy failed: ";
    }
  };
  const auto &CheckComplishAndDestroy = [&](cudaEvent_t event) -> bool {
    if (cudaEventQuery(event) == cudaSuccess) {
      DestroyEvent(event);
      return true;
    }
    if (cudaEventQuery(event) == cudaErrorNotReady) {
      LOG(ERROR) << "event is not completed or when destroying event pool.";
      return false;
    }
    LOG(ERROR) << "failed on cudaEventQuery when destroying event pool.";
    return false;
  };
  std::unique_lock<std::mutex> lock(mtx_);
  while (!incomplished_events_.empty()) {
    cudaEvent_t event = incomplished_events_.front();
    if (!CheckComplishAndDestroy(event)) {
      LOG(ERROR) << "failed on destroying event when destroying event pool.";
    }
    incomplished_events_.pop();
  }
}

cudaEvent_t EventPool::CreateCudaEventFromPool() {
  std::unique_lock<std::mutex> lock(mtx_);

  const auto &CreateNewEvent = [&]() -> cudaEvent_t {
    cudaEvent_t new_event;
    CUDA_CHECK(cudaEventCreate(&new_event));
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
}  // namespace deep_ep::detail
