// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/profiler/host_tracer.h"

#include <sstream>

#include "glog/logging.h"
#include "paddle/fluid/platform/flags.h"
#include "paddle/fluid/platform/profiler/common_event.h"
#include "paddle/fluid/platform/profiler/host_event_recorder.h"

// Used to filter events, works like glog VLOG(level).
// RecordEvent will works if host_trace_level >= level.
PADDLE_DEFINE_EXPORTED_int64(host_trace_level,
                             1,
                             "RecordEvent will works "
                             "if host_trace_level >= level.");

namespace paddle {
namespace platform {

namespace {

void ProcessHostEvents(const HostEventSection<CommonEvent>& host_events,
                       TraceEventCollector* collector) {
  for (const auto& thr_sec : host_events.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
    if (thr_sec.thread_name != kDefaultThreadName) {
      collector->AddThreadName(tid, thr_sec.thread_name);
    }
    for (const auto& evt : thr_sec.events) {
      HostTraceEvent event;
      event.name = evt.name;
      event.type = evt.type;
      event.start_ns = evt.start_ns;
      event.end_ns = evt.end_ns;
      event.process_id = host_events.process_id;
      event.thread_id = tid;
      collector->AddHostEvent(std::move(event));
    }
  }
}

void ProcessHostMemEvents(
    const HostEventSection<CommonMemEvent>& host_mem_events,
    TraceEventCollector* collector) {
  for (const auto& thr_sec : host_mem_events.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
    if (thr_sec.thread_name != kDefaultThreadName) {
      collector->AddThreadName(tid, thr_sec.thread_name);
    }
    for (const auto& evt : thr_sec.events) {
      MemTraceEvent event;
      event.timestamp_ns = evt.timestamp_ns;
      event.addr = evt.addr;
      event.type = evt.type;
      event.increase_bytes = evt.increase_bytes;
      event.place = evt.place.DebugString();
      event.current_allocated = evt.current_allocated;
      event.current_reserved = evt.current_reserved;
      event.peak_allocated = evt.peak_allocated;
      event.peak_reserved = evt.peak_reserved;
      event.process_id = host_mem_events.process_id;
      event.thread_id = tid;
      collector->AddMemEvent(std::move(event));
    }
  }
}

void ProcessOperatorSupplementEvents(
    const HostEventSection<OperatorSupplementOriginEvent>& op_supplement_events,
    TraceEventCollector* collector) {
  for (const auto& thr_sec : op_supplement_events.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
    if (thr_sec.thread_name != kDefaultThreadName) {
      collector->AddThreadName(tid, thr_sec.thread_name);
    }
    for (const auto& evt : thr_sec.events) {
      OperatorSupplementEvent event;
      event.timestamp_ns = evt.timestamp_ns;
      event.op_type = evt.op_type;
      std::map<std::string, std::vector<std::vector<int64_t>>> input_shapes;
      std::map<std::string, std::vector<std::string>> dtypes;
      std::string callstack;
      for (auto it = evt.input_shapes.begin(); it != evt.input_shapes.end();
           it++) {
        for (auto idx = 0lu; idx < it->second.size(); idx++) {
          input_shapes[it->first].push_back(std::vector<int64_t>());
          for (auto dim_idx = 0; dim_idx < it->second.at(idx).size();
               dim_idx++) {
            input_shapes[it->first][idx].push_back(
                it->second.at(idx).at(dim_idx));
          }
        }
      }
      for (auto it = evt.dtypes.begin(); it != evt.dtypes.end(); it++) {
        for (auto idx = 0lu; idx < it->second.size(); idx++) {
          dtypes[it->first].push_back(
              framework::proto::VarType::Type_Name(it->second.at(idx)));
        }
      }

      std::ostringstream result_string;
      for (auto it = evt.callstack.begin(); it != evt.callstack.end(); it++) {
        result_string << (*it) << std::endl;
      }
      event.input_shapes = input_shapes;
      event.dtypes = dtypes;
      event.callstack = result_string.str();
      event.process_id = op_supplement_events.process_id;
      event.thread_id = tid;
      collector->AddOperatorSupplementEvent(std::move(event));
    }
  }
}

}  // namespace

void HostTracer::PrepareTracing() {
  // warm up
  HostTraceLevel::GetInstance().SetLevel(options_.trace_level);
  state_ = TracerState::READY;
}

void HostTracer::StartTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::READY || state_ == TracerState::STOPED,
      true,
      platform::errors::PreconditionNotMet("TracerState must be READY"));
  HostEventRecorder<CommonEvent>::GetInstance().GatherEvents();
  HostEventRecorder<CommonMemEvent>::GetInstance().GatherEvents();
  HostEventRecorder<OperatorSupplementOriginEvent>::GetInstance()
      .GatherEvents();
  HostTraceLevel::GetInstance().SetLevel(options_.trace_level);
  state_ = TracerState::STARTED;
}

void HostTracer::StopTracing() {
  PADDLE_ENFORCE_EQ(
      state_,
      TracerState::STARTED,
      platform::errors::PreconditionNotMet("TracerState must be STARTED"));
  HostTraceLevel::GetInstance().SetLevel(HostTraceLevel::kDisabled);
  state_ = TracerState::STOPED;
}

void HostTracer::CollectTraceData(TraceEventCollector* collector) {
  PADDLE_ENFORCE_EQ(
      state_,
      TracerState::STOPED,
      platform::errors::PreconditionNotMet("TracerState must be STOPED"));
  HostEventSection<CommonEvent> host_events =
      HostEventRecorder<CommonEvent>::GetInstance().GatherEvents();
  ProcessHostEvents(host_events, collector);
  HostEventSection<CommonMemEvent> host_mem_events =
      HostEventRecorder<CommonMemEvent>::GetInstance().GatherEvents();
  ProcessHostMemEvents(host_mem_events, collector);
  HostEventSection<OperatorSupplementOriginEvent> op_supplement_events =
      HostEventRecorder<OperatorSupplementOriginEvent>::GetInstance()
          .GatherEvents();
  ProcessOperatorSupplementEvents(op_supplement_events, collector);
}

}  // namespace platform
}  // namespace paddle
