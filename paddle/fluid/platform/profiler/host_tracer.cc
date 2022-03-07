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
#include "glog/logging.h"
#include "paddle/fluid/platform/flags.h"
#include "paddle/fluid/platform/profiler/common_event.h"
#include "paddle/fluid/platform/profiler/host_event_recorder.h"

// Used to filter events, works like glog VLOG(level).
// RecordEvent will works if host_trace_level >= level.
PADDLE_DEFINE_EXPORTED_int64(host_trace_level, 2,
                             "RecordEvent will works "
                             "if host_trace_level >= level.");

namespace paddle {
namespace platform {

namespace {

void ProcessHostEvents(const HostEventSection& host_events,
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

}  // namespace

void HostTracer::PrepareTracing() {
  // warm up
  HostTraceLevel::GetInstance().SetLevel(options_.trace_level);
  state_ = TracerState::READY;
}

void HostTracer::StartTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::READY || state_ == TracerState::STOPED, true,
      platform::errors::PreconditionNotMet("TracerState must be READY"));
  HostEventRecorder::GetInstance().GatherEvents();
  HostTraceLevel::GetInstance().SetLevel(options_.trace_level);
  state_ = TracerState::STARTED;
}

void HostTracer::StopTracing() {
  PADDLE_ENFORCE_EQ(
      state_, TracerState::STARTED,
      platform::errors::PreconditionNotMet("TracerState must be STARTED"));
  HostTraceLevel::GetInstance().SetLevel(HostTraceLevel::kDisabled);
  state_ = TracerState::STOPED;
}

void HostTracer::CollectTraceData(TraceEventCollector* collector) {
  PADDLE_ENFORCE_EQ(
      state_, TracerState::STOPED,
      platform::errors::PreconditionNotMet("TracerState must be STOPED"));
  HostEventSection host_events =
      HostEventRecorder::GetInstance().GatherEvents();
  ProcessHostEvents(host_events, collector);
}

}  // namespace platform
}  // namespace paddle
