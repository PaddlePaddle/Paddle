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
// #include "paddle/fluid/platform/profiler_helper.h"

namespace paddle {

namespace platform {

extern std::mutex profiler_mu;
// extern static ProfilerState g_state;

void PrintMemProfiler(
    const std::map<Place, std::unordered_map<std::string, MemoryProfierReport>>&
        annotation_report,
    const size_t name_width, const size_t data_width);

void SynchronizeAllDevice();
void DealWithShowName();
void ParseEvents(const std::vector<std::vector<Event>>& events,
                 bool merge_thread,
                 EventSortingKey sorted_by = EventSortingKey::kDefault);
}  // namespace platform

namespace framework {

using paddle::framework::CostModel;
using paddle::framework::CostData;

CostData CostModel::ProfileMeasure(const std::string& device) {
  // if (fetch_cost_list == None) fetch_cost_list = {"time", "memory"};
  CostData cost_data;
  VLOG(2) << "Printing from Cost Model ";
  // VLOG(10)
  // <<paddle::platform::g_enable_nvprof_hook<<paddle::platform::should_send_profile_state<<paddle::platform::profiler_lister_id;
  SynchronizeAllDevice();

  MemEvenRecorder::Instance().Flush();

  std::lock_guard<std::mutex> l(profiler_mu);
  // if (g_state == ProfilerState::kDisabled) return cost_data;
  // Mark the profiling stop.
  Mark("_stop_profiler_");
  DealWithShowName();
  // VLOG(2) << "22222222";

  DeviceTracer* tracer = GetDeviceTracer();
  if (tracer->IsEnabled()) {
    tracer->Disable();
    tracer->GenEventKernelCudaElapsedTime();
    // tracer->GenProfile(profile_path);
  }

  std::vector<std::vector<Event>> all_events = GetAllEvents();

  VLOG(2) << "GetAllEvents, all events lenth:" << all_events.size()
          << all_events[0].size();
  ParseEvents(all_events, true);
  ParseEvents(all_events, false);
  // std::vector<std::vector<MemEvent>> all_mem_events = GetMemEvents();
  // VLOG(2) << "GetAllMemEvents, all mem events lenth:" <<
  // all_mem_events.size();

  // ResetProfiler();
  return cost_data;
}

}  // namespace framework
}  // namespace paddle
