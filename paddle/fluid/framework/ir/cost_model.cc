// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/cost_model.h"
#include "paddle/fluid/platform/device_tracer.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler_helper.h"

// std::vector<float> StopCostModel(std::vector<string>) {
//   SynchronizeAllDevice();
//   MemEvenRecorder::Instance().Flush();

//   std::lock_guard<std::mutex> l(profiler_mu);
//   if (g_state == ProfilerState::kDisabled) return;
//   // Mark the profiling stop.
//   Mark("_stop_profiler_");
//   DealWithShowName();

//   DeviceTracer *tracer = GetDeviceTracer();
//   if (tracer->IsEnabled()) {
//     tracer->Disable();
//     tracer->GenEventKernelCudaElapsedTime();
//   }

//   std::vector<std::vector<Event>> all_events = GetAllEvents();

//   std::vector<float> cost_list;
//   cost_list.push_back(GetTimeCost(all_events));

//   std::vector<std::vector<MemEvent>> all_mem_events = GetMemEvents();
//   cost_list.push_back(GetMemoryCost(all_mem_events));
//   ResetProfiler();
// }

// float GetTimeCost(const std::vector<std::vector<MemEvent>> &events) {
//   if (g_state == ProfilerState::kDisabled) return;

//   const std::vector<std::vector<Event>> *analyze_events;
//   std::vector<std::vector<Event>> merged_events_list;

//   std::vector<Event> merged_events;
//   for (size_t i = 0; i < events.size(); ++i) {
//     for (size_t j = 0; j < events[i].size(); ++j) {
//       merged_events.push_back(events[i][j]);
//     }
//   }
//   merged_events_list.push_back(merged_events);
//   analyze_events = &merged_events_list;

//   std::vector<std::vector<EventItem>> events_table;
//   std::multimap<std::string, EventItem> child_map;

//   size_t max_name_width = 0;
//   OverHead overhead;

//   std::string sorted_domain;
//   std::function<bool(const EventItem &, const EventItem &)> sorted_func;
//   EventSortingKey sorted_by = EventSortingKey::kDefault sorted_func =
//       SetSortedFunc(sorted_by, &sorted_domain);

//   AnalyzeEvent(analyze_events, &events_table, &child_map, sorted_func,
//                sorted_by, &max_name_width, &overhead, merge_thread);

//   float time_cost = GetCostFromEventTable(
//       events_table, child_map, sorted_func, sorted_by, overhead,
//       sorted_domain,
//       max_name_width + 8, 12, merge_thread, 0);
//   return time cost;
// }
// float GetMemoryCost(const std::vector<std::vector<MemEvent>> &events) {}

// float GetCostFromEventTable(float GetCostFromEventTable(
//     const std::vector<std::vector<EventItem>> &events_table,
//     const std::multimap<std::string, EventItem> &child_map,
//     std::function<bool(const EventItem &, const EventItem &)> sorted_func,
//     EventSortingKey sorted_by, const OverHead &overhead,
//     const std::string &sorted_domain, const size_t name_width,
//     const size_t data_width, bool merge_thread, int print_depth) {
//     }

CostModel::ProfileMeasure(Program* program,
                          std::vector<std::string> fetch_cost_list) {
  if (fetch_cost_list == None) fetch_cost_list = {"time", "memory"};
  CostData cost_data();
  SynchronizeAllDevice();
  MemEvenRecorder::Instance().Flush();
  if (should_send_profile_state&& profiler_lister_id&& g_enable_nvprof_hook =
          None) {
    int i = 1;
  }

  std::lock_guard<std::mutex> l(profiler_mu);
  if (g_state == ProfilerState::kDisabled) return;
  // Mark the profiling stop.
  Mark("_stop_profiler_");
  DealWithShowName();

  DeviceTracer* tracer = GetDeviceTracer();
  if (tracer->IsEnabled()) {
    tracer->Disable();
    tracer->GenEventKernelCudaElapsedTime();
    tracer->GenProfile(profile_path);
  }

  std::vector<std::vector<Event>> all_events = GetAllEvents();

  VLOG(2) << "GetAllEvents, all events lenth:" << all_events.size();
  std::vector<std::vector<MemEvent>> all_mem_events = GetMemEvents();
  VLOG(2) << "GetAllMemEvents, all mem events lenth:" << all_mem_events.size();

  ResetProfiler();
  g_state = ProfilerState::kDisabled;
  g_tracer_option = TracerOption::kDefault;
  should_send_profile_state = true;
}
