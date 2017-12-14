/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <forward_list>
#include <list>
#include <mutex>
#include <vector>
#include "paddle/platform/device_context.h"

namespace paddle {
namespace platform {

enum EventKind { kMark, kPushRange, kPopRange };

inline uint64_t GetTimeInNsec() {
  // using std::chrono;
  using clock = std::conditional<std::chrono::high_resolution_clock::is_steady,
                                 std::chrono::high_resolution_clock,
                                 std::chrono::steady_clock>::type;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
      .count();
}

class Event {
 public:
  // the DeviceContext is used to get the cuda stream.
  Event(EventKind kind, std::string name, uint32_t thread_id,
        const platform::DeviceContext* dev_ctx = nullptr)
      : kind_(kind), name_(std::move(name)), thread_id_(thread_id) {
    has_cuda_ = false;
#ifdef PADDLE_WITH_CUDA
    auto* cuda_dev_ctx =
        static_cast<const platform::CUDADeviceContext*>(dev_ctx);
    if (cuda_dev_ctx) {
      PADDLE_ENFORCE(cudaGetDevice(&device_));
      PADDLE_ENFORCE(cudaEventCreate(&event_));
      auto stream = cuda_dev_ctx->stream();
      PADDLE_ENFORCE(cudaEventRecord(event_, stream));
      has_cuda_ = true;
    }
#endif
    cpu_ns_ = GetTimeInNsec();
  }

  std::string kind() const {
    switch (kind_) {
      case EventKind::kMark:
        return "mark";
      case EventKind::kPushRange:
        return "push";
      case EventKind::kPopRange:
        return "pop";
    }
    PADDLE_THROW("Unknown EventKind.");
  }

  std::string name() const { return name_; }

  bool has_cuda() const { return has_cuda_; }

#ifdef PADDLE_WITH_CUDA
  cudaEvent_t event() const { return event_; }

  int device() const { return device_; }
#endif

  double CpuElapsedUs(const Event& e) const {
    return (e.cpu_ns_ - cpu_ns_) / (1000.0);
  }

  double CudaElapsedUs(const Event& e) const {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE(e.has_cuda() && has_cuda());
    PADDLE_ENFORCE(e.device() == device());
    PADDLE_ENFORCE(cudaEventSynchronize(event_));
    PADDLE_ENFORCE(cudaEventSynchronize(e.event()));
    float ms;
    PADDLE_ENFORCE(cudaEventElapsedTime(&ms, event_, e.event()));
    return ms * 1000.0;
#else
    PADDLE_THROW("CUDA is not enabled");
#endif
  }

 private:
  EventKind kind_;
  std::string name_;
  uint32_t thread_id_;
  int64_t cpu_ns_;
  bool has_cuda_;
#ifdef PADDLE_WITH_CUDA
  cudaEvent_t event_ = nullptr;
  int device_ = -1;
#endif
};

struct EventList {
  constexpr static std::size_t kMB = 1024 * 1024;
  constexpr static std::size_t kEventBlockSize = 16 * kMB;
  constexpr static std::size_t kEventSize = sizeof(Event);
  constexpr static std::size_t kEventAlign = alignof(Event);
  constexpr static std::size_t kNumBlock =
      kEventBlockSize /
      ((kEventSize + kEventAlign - 1) / kEventAlign * kEventAlign);

  template <typename... Args>
  void Record(Args&&... args) {
    if (event_blocks.empty() || event_blocks.front().size() == kNumBlock) {
      event_blocks.emplace_front();
      event_blocks.front().reserve(kNumBlock);
    }
    event_blocks.front().emplace_back(std::forward<Args>(args)...);
  }

  std::vector<Event> Reduce() {
    std::vector<Event> result;
    for (auto& block : event_blocks) {
      result.insert(result.begin(), std::make_move_iterator(block.begin()),
                    std::make_move_iterator(block.end()));
    }
    event_blocks.clear();
    return result;
  }

  std::forward_list<std::vector<Event>> event_blocks;
};

enum ProfilerState {
  kDisabled,
  kCPU,
  kCUDA,
};

// The profiler state, the initial value is ProfilerState::kDisabled
extern ProfilerState kState;
// The global mutex
extern std::mutex kAllEventListsMutex;
// The total event lists of all threads
extern std::list<std::shared_ptr<EventList>> kAllEventLists;
// The thread local event list only can be accessed by the specific thread
extern thread_local std::shared_ptr<EventList> kEventList;
// The thread index of each thread
extern thread_local int32_t kThreadId;
// The kNextThreadId is a global counter for threads, by the kThreadId and
// kNextThreadId, we can know how many threads have created EventList.
extern uint32_t kNextThreadId;

inline EventList& GetEventList() {
  if (!kEventList) {
    std::lock_guard<std::mutex> guard(kAllEventListsMutex);
    kEventList = std::make_shared<EventList>();
    kThreadId = kNextThreadId++;
    kAllEventLists.emplace_front(kEventList);
  }
  return *kEventList;
}

inline void Mark(const std::string name,
                 const platform::DeviceContext* dev_ctx = nullptr) {
  GetEventList().Record(EventKind::kMark, std::move(name), kThreadId, dev_ctx);
}

struct RecordEvent {
  explicit RecordEvent(const std::string name,
                       platform::DeviceContext* dev_ctx = nullptr) {
    if (kState == ProfilerState::kDisabled) return;
    dev_ctx_ = dev_ctx;
    GetEventList().Record(EventKind::kPushRange, std::move(name), kThreadId,
                          dev_ctx_);
  }

  ~RecordEvent() {
    if (kState == ProfilerState::kDisabled) return;
    GetEventList().Record(EventKind::kPopRange, std::string(), kThreadId,
                          dev_ctx_);
  }
  platform::DeviceContext* dev_ctx_;
};

void EnableProfiler(ProfilerState state);
std::vector<std::vector<Event>> DisableProfiler();

}  // namespace platform
}  // namespace paddle
