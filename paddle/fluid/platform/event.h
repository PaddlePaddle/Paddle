/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <string>

namespace paddle {
namespace platform {

enum EventType { kMark, kPushRange, kPopRange };

class Event {
 public:
  // The DeviceContext is used to get the cuda stream.
  // If CPU profiling mode, can pass nullptr.
  Event(EventType type, std::string name, uint32_t thread_id);

  const EventType& type() const;
  std::string name() const { return name_; }
  uint32_t thread_id() const { return thread_id_; }

#ifdef PADDLE_WITH_CUDA
#ifndef PADDLE_WITH_CUPTI
  cudaEvent_t event() const { return event_; }
  int device() const { return device_; }
#endif
#endif

  double CpuElapsedMs(const Event& e) const;
  double CudaElapsedMs(const Event& e) const;

 private:
  EventType type_;
  std::string name_;
  uint32_t thread_id_;
  int64_t cpu_ns_;
#ifdef PADDLE_WITH_CUDA
#ifdef PADDLE_WITH_CUPTI
  int64_t gpu_ns_ = 0;

 public:
  void AddCudaElapsedTime(int64_t start_ns, int64_t end_ns) {
    gpu_ns_ += end_ns - start_ns;
  }

 private:
#else
  cudaEvent_t event_ = nullptr;
  int device_ = -1;
#endif
#endif
};
}  // namespace platform
}  // namespace paddle
