/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <random>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/phi/api/profiler/device_tracer.h"

namespace phi {

class ProfilerHelper {
 public:
  // The profiler state, the initial value is ProfilerState::kDisabled
  static ProfilerState g_state;
  // To hook RecordEvent's events, use it to nvtx timeline
  static bool g_enable_nvprof_hook;
  // The thread local event list only can be accessed by the specific thread
  // The thread index of each thread
  static thread_local uint64_t g_thread_id;
  // The g_next_thread_id is a global counter for threads, by the g_thread_id
  // and g_next_thread_id, we can know how many threads have created EventList.
  static uint32_t g_next_thread_id;
  // The global mutex
  static std::mutex g_all_event_lists_mutex;
  // The total event lists of all threads
  static std::list<std::shared_ptr<EventList<Event>>> g_all_event_lists;
  // The thread local event list only can be accessed by the specific thread
  static thread_local std::shared_ptr<EventList<Event>> g_event_list;

  static std::list<std::shared_ptr<EventList<MemEvent>>> g_all_mem_event_lists;
  static thread_local std::shared_ptr<EventList<MemEvent>> g_mem_event_list;
  static std::mutex g_all_mem_event_lists_mutex;
};

inline uint64_t GetTimeInNsec() {
  using clock = std::conditional<std::chrono::high_resolution_clock::is_steady,
                                 std::chrono::high_resolution_clock,
                                 std::chrono::steady_clock>::type;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
      .count();
}

inline EventList<Event> &GetEventList() {
  if (!ProfilerHelper::g_event_list) {
    std::lock_guard<std::mutex> guard(ProfilerHelper::g_all_event_lists_mutex);
    ProfilerHelper::g_event_list = std::make_shared<EventList<Event>>();
    ProfilerHelper::g_thread_id = ProfilerHelper::g_next_thread_id++;
    ProfilerHelper::g_all_event_lists.emplace_front(
        ProfilerHelper::g_event_list);
    RecoreCurThreadId(ProfilerHelper::g_thread_id);
  }
  return *ProfilerHelper::g_event_list;
}

}  // namespace phi
