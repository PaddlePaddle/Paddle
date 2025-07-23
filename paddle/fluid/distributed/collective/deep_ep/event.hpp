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

// The file has been adapted from DeepSeek DeepEP project
// Copyright (c) 2025 DeepSeek
// Licensed under the MIT License -
// https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

#pragma once

#include <memory>

#include "paddle/fluid/distributed/collective/deep_ep/include/CUDAStream.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/event.h"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"

namespace deep_ep {

struct EventHandle {
  std::shared_ptr<deep_ep::detail::Event> event;

  EventHandle() {
    event = std::make_shared<deep_ep::detail::Event>();
    // LOG(WARNING) << "EventHandle constructor is called without record current
    // stream";
    event->record(deep_ep::detail::getCurrentCUDAStream().raw_stream());
  }

  void CalcStreamWait(int context_ring_id) const;
  void CommStreamWait(int context_ring_id) const;

  explicit EventHandle(const cudaStream_t& stream) {
    event = std::make_shared<deep_ep::detail::Event>();
    event->record(stream);
  }

  explicit EventHandle(const phi::CUDAStream& stream) {
    event = std::make_shared<deep_ep::detail::Event>();
    event->record(stream.raw_stream());
  }

  EventHandle(const EventHandle& other) = default;

  void current_stream_wait() const {
    CUDA_CHECK(cudaStreamWaitEvent(
        deep_ep::detail::getCurrentCUDAStream().raw_stream(),
        event->cuda_event(),
        0));
  }
};

EventHandle GetEventHandleFromCalcStream(int context_ring_id);
EventHandle GetEventHandleFromCommStream(int context_ring_id);
EventHandle GetEventHandleFromCustomStream(const phi::CUDAStream& stream);

inline deep_ep::detail::Event create_event(const cudaStream_t& s) {
  auto event = deep_ep::detail::Event();
  event.record(s);
  return event;
}

inline void stream_wait(const cudaStream_t& s_0, const cudaStream_t& s_1) {
  EP_HOST_ASSERT(s_0 != s_1);
  CUDA_CHECK(cudaStreamWaitEvent(s_0, create_event(s_1).cuda_event(), 0));
}

inline void stream_wait(const cudaStream_t& s, const EventHandle& event) {
  CUDA_CHECK(cudaStreamWaitEvent(s, event.event->cuda_event(), 0));
}

}  // namespace deep_ep
