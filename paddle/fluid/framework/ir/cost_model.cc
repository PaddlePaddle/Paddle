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

#include <memory>

#include "paddle/common/errors.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/place.h"

namespace paddle::framework {

using ir::Graph;
using phi::Event;
using phi::MemEvent;

const double CostData::NOT_MEASURED = -1;

CostData::~CostData() {  // NOLINT
  // TODO(zhhsplendid): when we save a copy of program/graph, we should delete
  // here.
}

double CostData::GetOpTimeMs(int op_id) const { return op_time_ms_.at(op_id); }
double CostData::GetOpMemoryBytes(int op_id) const {
  return op_memory_bytes_.at(op_id);
}
double CostData::GetWholeTimeMs() const { return whole_time_ms_; }
double CostData::GetWholeMemoryBytes() const { return whole_memory_bytes_; }

const Graph* CostData::GetGraph() const { return graph_; }
const ProgramDesc* CostData::GetProgram() const { return program_; }

static bool StringHasEnding(const std::string& full,
                            const std::string& ending) {
  if (full.length() < ending.length()) {
    return false;
  }
  if (full.length() == ending.length()) {
    return full == ending;
  }
  size_t prefix_len = full.length() - ending.length();
  return 0 == full.compare(prefix_len, ending.length(), ending) &&
         full[prefix_len - 1] == '/';
}

bool CostData::SetCostData(const ProgramDesc& program,
                           const std::vector<std::vector<Event>>& time_events) {
  // TODO(zhhsplendid): Make a copy so that CostData can be available even if
  // SWE changes Program, the copy can be saved into pointer program_
  if (program.Size() == 0) {
    whole_time_ms_ = 0;
    whole_memory_bytes_ = 0;
    return true;
  }

  if (time_events.empty()) {
    LOG(WARNING) << "Input time_events for CostModel is empty";
    return false;
  }

  std::vector<Event> main_thread_events = time_events[0];
  // Support global block only
  // TODO(zhhsplendid): support sub blocks
  const BlockDesc& global_block = program.Block(0);
  size_t op_size = global_block.OpSize();
  if (op_size == 0) {
    whole_time_ms_ = 0;
    whole_memory_bytes_ = 0;
    return true;
  }

  bool event_to_cost_success = true;
  size_t event_index = 0;
  for (size_t i = 0; i < op_size; ++i) {
    const OpDesc* op_desc = global_block.Op(static_cast<int>(i));
    std::string op_type = op_desc->Type();

    while (event_index < main_thread_events.size()) {
      if (StringHasEnding(main_thread_events[event_index].name(), op_type) &&
          main_thread_events[event_index].type() ==
              phi::EventType::kPushRange) {
        break;
      }
      ++event_index;
    }
    if (event_index >= main_thread_events.size()) {
      LOG(WARNING) << "Input time_events for Op " << i << ", type '" << op_type
                   << "' have wrong format, skip this Op.";
      event_to_cost_success = false;
      continue;
    }
    size_t op_push_index = event_index;

    while (event_index < main_thread_events.size()) {
      // Is it possible to Push a lot of Ops with same type and then Pop?
      // ControlFlow Op can be like that, but this version only support global
      // block
      // TODO(zhhsplendid): make a more strict mapping between push and pop
      if (StringHasEnding(main_thread_events[event_index].name(), op_type) &&
          main_thread_events[event_index].type() == phi::EventType::kPopRange) {
        break;
      }
      ++event_index;
    }
    if (event_index >= main_thread_events.size()) {
      LOG(WARNING) << "Input time_events for Op " << i << ", type '" << op_type
                   << "' have wrong format, skip this Op.";
      event_to_cost_success = false;
      continue;
    }
    size_t op_pop_index = event_index;
    double cpu_time_ms = main_thread_events[op_push_index].CpuElapsedMs(
        main_thread_events[op_pop_index]);
    double gpu_time_ms = 0;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    gpu_time_ms = main_thread_events[op_push_index].CudaElapsedMs(
        main_thread_events[op_pop_index]);
#endif
    double time_ms = gpu_time_ms + cpu_time_ms;
    op_time_ms_[static_cast<int>(i)] = time_ms;
  }

  event_index = 0;
  int start_profiler_idx = -1;
  int stop_profiler_idx = -1;
  while (event_index < main_thread_events.size()) {
    if (main_thread_events[event_index].name() == "_start_profiler_") {
      start_profiler_idx = static_cast<int>(event_index);
    } else if (main_thread_events[event_index].name() == "_stop_profiler_") {
      stop_profiler_idx = static_cast<int>(event_index);
      break;
    }
    ++event_index;
  }
  if (start_profiler_idx != -1 && stop_profiler_idx != -1) {
    double cpu_time_ms = main_thread_events[start_profiler_idx].CpuElapsedMs(
        main_thread_events[stop_profiler_idx]);
    double gpu_time_ms = 0;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    gpu_time_ms = main_thread_events[start_profiler_idx].CudaElapsedMs(
        main_thread_events[stop_profiler_idx]);
#endif
    whole_time_ms_ = gpu_time_ms + cpu_time_ms;
  } else {
    LOG(WARNING) << "Input time_events for whole time have wrong format";
    event_to_cost_success = false;
  }

  return event_to_cost_success;
}

void PrintEvents(const std::vector<std::vector<Event>>* time_events,
                 const std::vector<std::vector<MemEvent>>* mem_events) {
  if (time_events != nullptr) {
    for (size_t i = 0; i < time_events->size(); ++i) {
      for (size_t j = 0; j < (*time_events)[i].size(); ++j) {
        VLOG(4) << "Print time event (" << i << ", " << j << ")" << std::endl;
        VLOG(4) << (*time_events)[i][j].name() << " "
                << (*time_events)[i][j].attr() << std::endl;
        VLOG(4) << "This: " << &(*time_events)[i][j]
                << ", Parent: " << (*time_events)[i][j].parent() << std::endl;
        if ((*time_events)[i][j].role() == phi::EventRole::kInnerOp) {
          VLOG(4) << "role kInnerOp" << std::endl;
        } else if ((*time_events)[i][j].role() == phi::EventRole::kUniqueOp) {
          VLOG(4) << "role kUniqueOp" << std::endl;
        } else if ((*time_events)[i][j].role() == phi::EventRole::kOrdinary) {
          VLOG(4) << "role kOrdinary" << std::endl;
        } else if ((*time_events)[i][j].role() == phi::EventRole::kSpecial) {
          VLOG(4) << "role kSpecial" << std::endl;
        }

        if ((*time_events)[i][j].type() == phi::EventType::kPopRange) {
          VLOG(4) << "type kPopRange" << std::endl;
        } else if ((*time_events)[i][j].type() == phi::EventType::kPushRange) {
          VLOG(4) << "type kPushRange" << std::endl;
        } else if ((*time_events)[i][j].type() == phi::EventType::kMark) {
          VLOG(4) << "type kMark" << std::endl;
        }
        VLOG(4) << std::endl;
      }
    }
  }
  if (mem_events != nullptr) {
    for (size_t i = 0; i < mem_events->size(); ++i) {
      for (size_t j = 0; j < (*mem_events)[i].size(); ++j) {
        VLOG(4) << "Print mem event (" << i << ", " << j << ")" << std::endl;
        VLOG(4) << (*mem_events)[i][j].annotation() << std::endl;
      }
    }
  }
}

std::string ToLowerCopy(const std::string& in) {
  std::string out(in);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  return out;
}

CostData CostModel::ProfileMeasure(
    const ProgramDesc& main_program,
    const ProgramDesc& startup_program,
    const std::string& device,
    const std::vector<std::string>& fetch_cost_list) const {
  // Currently fetch_cost_list is useless
  // TODO(zhhsplendid): support different fetch data

  platform::ProfilerState profiler_state;
  phi::Place place;

  std::string device_lower_case = ToLowerCopy(device);
  if (device_lower_case == "cpu") {
    profiler_state = platform::ProfilerState::kCPU;
    place = phi::CPUPlace();
  } else if (device_lower_case == "gpu") {
    profiler_state = platform::ProfilerState::kAll;
    place = phi::GPUPlace();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Not support %s in CostModel now", device));
  }

  Executor executor(place);
  Scope scope;
  executor.Run(startup_program, &scope, /*block_id = */ 0);

  // TODO(zhhsplendid): handle the case that Profiler is already enabled
  platform::SetTracerOption(platform::TracerOption::kAllOpDetail);
  platform::EnableProfiler(profiler_state);
  executor.Run(main_program, &scope, /*block_id = */ 0);

  std::unique_ptr<std::vector<std::vector<Event>>> time_events(
      new std::vector<std::vector<Event>>());
  std::unique_ptr<std::vector<std::vector<MemEvent>>> mem_events(
      new std::vector<std::vector<MemEvent>>());

  platform::CompleteProfilerEvents(
      /*tracer_profile= */ nullptr, time_events.get(), mem_events.get());

  // TODO(zhhsplendid): remove debug vlog after this series of work
  PrintEvents(time_events.get(), mem_events.get());

  // Convert events to cost data
  CostData cost_data;
  cost_data.SetCostData(main_program, *time_events);

  return cost_data;
}

}  // namespace paddle::framework
