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
#include <map>

namespace paddle {
namespace platform {

ProfilerState kState = ProfilerState::kDisabled;
uint32_t kNextThreadId = 0;
std::mutex kAllEventListsMutex;
std::list<std::shared_ptr<EventList>> kAllEventLists;
thread_local std::shared_ptr<EventList> kEventList;
thread_local int32_t kThreadId;

void EnableProfiler(ProfilerState state) {
  PADDLE_ENFORCE(state != ProfilerState::kDisabled,
                 "Can't enbale profling, since the input state is ",
                 "ProfilerState::kDisabled");
  PADDLE_ENFORCE(kState == ProfilerState::kDisabled,
                 "The profiling state should be disabled when calling ",
                 "EnableProfiler.");
  kState = state;
#ifdef PADDLE_WITH_CUDA
  auto ForEachDevice = [](std::function<void(int)> op) {
    int count = GetCUDADeviceCount();
    for (int i = 0; i < count; i++) {
      DeviceGuard dev_guard(i);
      op(i);
    }
  };
  if (kState == ProfilerState::kCUDA) {
    // Generate some dummy evenets first to reduce the startup overhead.
    for (int i = 0; i < 5; i++) {
      ForEachDevice([](int d) {
        DeviceContext* dev_ctx = new CUDADeviceContext(GPUPlace(d));
        Mark("_cuda_startup_", dev_ctx);
        dev_ctx->Wait();
      });
    }
  }
#endif
  // Mark the profiling start.
  Mark("_start_profiler_");
}

std::vector<std::vector<Event>> DisableProfiler() {
  PADDLE_ENFORCE(kState != ProfilerState::kDisabled,
                 "Can't disable profiling, since it's not starting.");
  // Mark the profiling stop.
  Mark("_stop_profiler_");
  kState = ProfilerState::kDisabled;
  std::vector<std::vector<Event>> result;
  std::lock_guard<std::mutex> guard(kAllEventListsMutex);
  for (auto it = kAllEventLists.begin(); it != kAllEventLists.end(); ++it) {
    auto& list = *it;
    result.emplace_back(list->Reduce());
  }
  return result;
}

void PushEvent(const std::string name, const platform::DeviceContext* dev_ctx) {
  GetEventList().Record(EventKind::kPushRange, std::move(name), kThreadId,
                        dev_ctx);
}

void PopEvent(const std::string name, const platform::DeviceContext* dev_ctx) {
  GetEventList().Record(EventKind::kPopRange, std::move(name), kThreadId,
                        dev_ctx);
}

void ParseEvents(std::vector<std::vector<Event>> events) {
  std::map<std::string, std::tuple<int, double, double>> events_table;
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
          Event pushed_event = *rit;
          double cpu_time = rit->CpuElapsedUs(events[i][j]);
          double cuda_time = 0;
#ifdef PADDLE_WITH_CUDA
          cuda_time = rit->CudaElapsedUs(events[i][j]);
#endif
          if (events_table.find(rit->name()) == events_table.end()) {
            events_table[rit->name()] = std::make_tuple(1, cpu_time, cuda_time);
          } else {
            std::get<0>(events_table[rit->name()]) += 1;
            std::get<1>(events_table[rit->name()]) += cpu_time;
            std::get<2>(events_table[rit->name()]) += cuda_time;
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
  std::cout << "\nEvents\t\tCalls\t\tTotal CPU time\t\tTotal GPU time\n";
  for (std::map<std::string, std::tuple<int, double, double>>::iterator it =
           events_table.begin();
       it != events_table.end(); ++it) {
    std::cout << it->first << "\t\t" << std::get<0>(it->second) << "\t\t"
              << std::get<1>(it->second) << "\t\t" << std::get<2>(it->second)
              << std::endl;
  }
}

}  // namespace platform
}  // namespace paddle
