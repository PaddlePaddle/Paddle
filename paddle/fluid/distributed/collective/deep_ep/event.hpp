// MIT License

// Copyright (c) 2025 DeepSeek

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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

  explicit EventHandle(const cudaStream_t& stream) {
    event = std::make_shared<deep_ep::detail::Event>();
    event->record(stream);
  }

  EventHandle(const EventHandle& other) = default;

  void current_stream_wait() const {
    CUDA_CHECK(cudaStreamWaitEvent(
        deep_ep::detail::getCurrentCUDAStream().raw_stream(),
        event->cuda_event(),
        0));
  }
};

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
