/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler.pb.h"

namespace paddle {
namespace platform {

enum EventKind { kMark, kPushRange, kPopRange };

class Event {
 public:
  // The DeviceContext is used to get the cuda stream.
  // If CPU profiling mode, can pass nullptr.
  Event(EventKind kind, std::string name, uint32_t thread_id,
        const DeviceContext* dev_ctx);

  std::string kind() const;
  std::string name() const { return name_; }
  uint32_t thread_id() const { return thread_id_; }
  bool has_cuda() const { return has_cuda_; }

#ifdef PADDLE_WITH_CUDA
  cudaEvent_t event() const { return event_; }
  int device() const { return device_; }
#endif

  double CpuElapsedMs(const Event& e) const;
  double CudaElapsedMs(const Event& e) const;

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
  constexpr static size_t kMB = 1024 * 1024;
  constexpr static size_t kEventBlockSize = 16 * kMB;
  constexpr static size_t kEventSize = sizeof(Event);
  constexpr static size_t kEventAlign = alignof(Event);
  constexpr static size_t kNumBlock =
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

  void Clear() { event_blocks.clear(); }

  std::forward_list<std::vector<Event>> event_blocks;
};

enum ProfilerState {
  kDisabled,  // disabled state
  kCPU,       // CPU profiling state
  kCUDA,      // GPU profiling state
  kAll,       // Profile both CPU and GPU. (Currently experimental).
};

void Mark(const std::string& name, const DeviceContext* dev_ctx);

void PushEvent(const std::string& name, const DeviceContext* dev_ctx);

void PopEvent(const std::string& name, const DeviceContext* dev_ctx);

struct RecordEvent {
  RecordEvent(const std::string& name, const DeviceContext* dev_ctx);

  ~RecordEvent();

  uint64_t start_ns_;
  // The device context is used by Event to get the current cuda stream.
  const DeviceContext* dev_ctx_;
  // Event name
  std::string name_;
  // Need to distinguish name by op type, block_id, program_id and perhaps
  // different kernel invocations within an op.
  std::string full_name_;
};

// Return the event list of all threads. Assumed the returned value calls
// event_lists, event_lists[i][j] represents the j-th Event of i-th thread.
std::vector<std::vector<Event>> GetAllEvents();

// The information of each event given in the profiling report
struct EventItem {
  std::string name;
  int calls;
  double total_time;
  double min_time;
  double max_time;
  double ave_time;
};

// Candidate keys to sort the profiling report
enum EventSortingKey { kDefault, kCalls, kTotal, kMin, kMax, kAve };

// Enable the profiling function.
void EnableProfiler(ProfilerState state);

// Clear the g_all_event_lists, which is total event lists of all threads.
void ResetProfiler();

void DisableProfiler(EventSortingKey sorted_key,
                     const std::string& profile_path);

// Parse the event list and output the profiling report
void ParseEvents(std::vector<std::vector<Event>>&,
                 EventSortingKey sorted_by = EventSortingKey::kDefault);

// Print results
void PrintProfiler(std::vector<std::vector<EventItem>>& events_table,
                   std::string& sorted_domain, const size_t name_width,
                   const size_t data_width);

}  // namespace platform
}  // namespace paddle
