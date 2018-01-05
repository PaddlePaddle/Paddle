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

#include "paddle/platform/profiler.h"
#include <iomanip>
#include <map>

namespace paddle {
namespace platform {

// The profiler state, the initial value is ProfilerState::kDisabled
static ProfilerState g_state = ProfilerState::kDisabled;
// The thread local event list only can be accessed by the specific thread
// The thread index of each thread
static thread_local int32_t g_thread_id;
// The g_next_thread_id is a global counter for threads, by the g_thread_id and
// g_next_thread_id, we can know how many threads have created EventList.
static uint32_t g_next_thread_id = 0;
// The global mutex
static std::mutex g_all_event_lists_mutex;
// The total event lists of all threads
static std::list<std::shared_ptr<EventList>> g_all_event_lists;
// The thread local event list only can be accessed by the specific thread
static thread_local std::shared_ptr<EventList> g_event_list;

inline uint64_t GetTimeInNsec() {
  using clock = std::conditional<std::chrono::high_resolution_clock::is_steady,
                                 std::chrono::high_resolution_clock,
                                 std::chrono::steady_clock>::type;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
      .count();
}

Event::Event(EventKind kind, std::string name, uint32_t thread_id,
             DeviceContext* dev_ctx)
    : kind_(kind),
      name_(std::move(name)),
      thread_id_(thread_id),
      has_cuda_(false) {
#ifdef PADDLE_WITH_CUDA
  auto* cuda_dev_ctx = static_cast<const CUDADeviceContext*>(dev_ctx);
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

std::string Event::kind() const {
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

double Event::CpuElapsedUs(const Event& e) const {
  return (e.cpu_ns_ - cpu_ns_) / (1000.0);
}

double Event::CudaElapsedUs(const Event& e) const {
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

#ifdef PADDLE_WITH_CUDA
static void ForEachDevice(std::function<void(int)> func) {
  auto original_device = GetCurrentDeviceId();
  int count = GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    SetDeviceId(i);
    func(i);
  }
  SetDeviceId(original_device);
}
#endif

inline EventList& GetEventList() {
  if (!g_event_list) {
    std::lock_guard<std::mutex> guard(g_all_event_lists_mutex);
    g_event_list = std::make_shared<EventList>();
    g_thread_id = g_next_thread_id++;
    g_all_event_lists.emplace_front(g_event_list);
  }
  return *g_event_list;
}

void Mark(const std::string& name, DeviceContext* dev_ctx) {
  GetEventList().Record(EventKind::kMark, std::move(name), g_thread_id,
                        dev_ctx);
}

RecordEvent::RecordEvent(const std::string& name, DeviceContext* dev_ctx) {
  if (g_state == ProfilerState::kDisabled) return;
  dev_ctx_ = dev_ctx;
  name_ = name;
  GetEventList().Record(EventKind::kPushRange, std::move(name), g_thread_id,
                        dev_ctx_);
}

RecordEvent::~RecordEvent() {
  if (g_state == ProfilerState::kDisabled) return;
  GetEventList().Record(EventKind::kPopRange, std::move(name_), g_thread_id,
                        dev_ctx_);
}

void EnableProfiler(ProfilerState state) {
  PADDLE_ENFORCE(state != ProfilerState::kDisabled,
                 "Can't enbale profling, since the input state is ",
                 "ProfilerState::kDisabled");
  PADDLE_ENFORCE(g_state == ProfilerState::kDisabled,
                 "The profiling state should be disabled when calling ",
                 "EnableProfiler.");
  g_state = state;
#ifdef PADDLE_WITH_CUDA
  if (g_state == ProfilerState::kCUDA) {
    // Generate some dummy evenets first to reduce the startup overhead.
    for (int i = 0; i < 5; i++) {
      ForEachDevice([](int d) {
        DeviceContext* dev_ctx = new CUDADeviceContext(CUDAPlace(d));
        Mark("_cuda_startup_", dev_ctx);
        dev_ctx->Wait();
      });
    }
  }
#endif
  // Mark the profiling start.
  Mark("_start_profiler_", nullptr);
}

std::vector<std::vector<Event>> DisableProfiler() {
  PADDLE_ENFORCE(g_state != ProfilerState::kDisabled,
                 "Can't disable profiling, since it's not starting.");
  // Mark the profiling stop.
  Mark("_stop_profiler_", nullptr);
  g_state = ProfilerState::kDisabled;
  std::vector<std::vector<Event>> result;
  std::lock_guard<std::mutex> guard(g_all_event_lists_mutex);
  for (auto it = g_all_event_lists.begin(); it != g_all_event_lists.end();
       ++it) {
    result.emplace_back((*it)->Reduce());
  }
  return result;
}

void PushEvent(const std::string& name, DeviceContext* dev_ctx) {
  GetEventList().Record(EventKind::kPushRange, std::move(name), g_thread_id,
                        dev_ctx);
}

void PopEvent(const std::string& name, DeviceContext* dev_ctx) {
  GetEventList().Record(EventKind::kPopRange, std::move(name), g_thread_id,
                        dev_ctx);
}

void ParseEvents(std::vector<std::vector<Event>>& events) {
  // Event name :: counts :: ave  ::  min   ::  max :: total
  std::map<std::string, std::tuple<int, double, double, double, double>>
      events_table;
  for (size_t i = 0; i < events.size(); i++) {
    std::list<Event> pushed_events;
    for (size_t j = 0; j < events[i].size(); j++) {
      if (events[i][j].kind() == "push") {
        pushed_events.push_back(events[i][j]);
      }
      if (events[i][j].kind() == "pop") {
        std::list<Event>::reverse_iterator rit = pushed_events.rbegin();
        while (rit->name() != events[i][j].name() &&
               rit != pushed_events.rend()) {
          ++rit;
        }
        if (rit != pushed_events.rend()) {
#ifdef PADDLE_WITH_CUDA
          double event_time = rit->CudaElapsedUs(events[i][j]);
#else
          double event_time = rit->CpuElapsedUs(events[i][j]);
#endif
          std::string event_name =
              "thread" + std::to_string(rit->thread_id()) + "::" + rit->name();
          if (events_table.find(event_name) == events_table.end()) {
            events_table[event_name] =
                std::make_tuple(1, event_time, event_time, event_time, 0);
          } else {
            std::get<0>(events_table[event_name]) += 1;
            // total time
            std::get<1>(events_table[event_name]) += event_time;
            // min time
            if (std::get<2>(events_table[event_name]) > event_time) {
              std::get<2>(events_table[event_name]) = event_time;
            }
            // max time
            if (std::get<3>(events_table[event_name]) < event_time) {
              std::get<3>(events_table[event_name]) = event_time;
            }
          }
          // remove the start marker from the list
          pushed_events.erase((++rit).base());
        } else {
          std::cout << "Warning: can not find the start marker of event "
                    << events[i][j].name();
        }
      }
    }
  }
  // output events table
  std::cout << std::setw(20) << "Events" << std::setw(10) << "Calls"
            << std::setw(10) << "Total" << std::setw(10) << "Min"
            << std::setw(10) << "Max" << std::setw(10) << "Ave" << std::endl;
  for (std::map<std::string,
                std::tuple<int, double, double, double, double>>::iterator it =
           events_table.begin();
       it != events_table.end(); ++it) {
    // average time
    std::get<4>(it->second) = std::get<1>(it->second) / std::get<0>(it->second);
    std::cout << std::setw(20) << it->first << std::setw(10)
              << std::get<0>(it->second) << std::setw(10)
              << std::get<1>(it->second) << std::setw(10)
              << std::get<2>(it->second) << std::setw(10)
              << std::get<3>(it->second) << std::setw(10)
              << std::get<4>(it->second) << std::endl;
  }
}

}  // namespace platform
}  // namespace paddle
