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
#include <set>
#include <unordered_map>
#include "glog/logging.h"
#include "paddle/fluid/platform/profiler/common_event.h"
#include "paddle/fluid/platform/profiler/host_event_recorder.h"

namespace paddle {
namespace platform {

namespace {

void ProcessHostEvents(const HostEventSection& host_events,
                       TraceEventCollector* collector) {
  for (const auto& thr_sec : host_events.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
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

DECLARE_bool(use_stream_safe_cuda_allocator);

void StatisticsHostEvents(const TraceEventCollector& collector) {
  std::unordered_map<std::string, int> name2idx;
  std::vector<size_t> idx2cnt;
  std::vector<uint64_t> idx2total_ns;
  std::vector<std::set<uint64_t>> idx2threads;
#define REG_EVENT(event)             \
  name2idx[#event] = idx2cnt.size(); \
  idx2cnt.push_back(0);              \
  idx2total_ns.push_back(0);         \
  idx2threads.push_back({})
  REG_EVENT(AutoGrowthBestFitAllocator::Allocate);
  REG_EVENT(AutoGrowthBestFitAllocator::Free);
  REG_EVENT(StreamSafeCUDAAllocator::Allocate);
  REG_EVENT(StreamSafeCUDAAllocator::Free);
  REG_EVENT(WorkQueue::AddTask);
  REG_EVENT(prepare_data);
  REG_EVENT(compute);
  REG_EVENT(Compute);
  REG_EVENT(ProfileStep);
#undef REG_EVENT

  // statistics
  for (const auto& evt : collector.HostEvents()) {
    VLOG(1) << "name: " << evt.name;
    std::string prefix = evt.name;
    size_t split = evt.name.find('#');
    if (split != std::string::npos) {
      prefix = evt.name.substr(0, split);
      VLOG(1) << "prefix name: " << prefix;
    }
    auto iter = name2idx.find(prefix);
    if (iter != name2idx.end()) {
      int idx = iter->second;
      ++idx2cnt[idx];
      idx2total_ns[idx] += evt.end_ns - evt.start_ns;
      idx2threads[idx].insert(evt.thread_id);
    }
  }

  VLOG(1) << "=========Events Statistics==========";
  for (const auto& kv : name2idx) {
    VLOG(1) << kv.first << " cnt:" << idx2cnt[kv.second]
            << " total(ns):" << idx2total_ns[kv.second]
            << " threads:" << idx2threads[kv.second].size();
  }

  VLOG(1) << "========Executor analysis========";
#define GET_EVENT_TATALTIME(event) idx2total_ns[name2idx[#event]]
#define GET_EVENT_COUNT(event) idx2cnt[name2idx[#event]]
  if (GET_EVENT_COUNT(compute) == 0) {
    VLOG(1) << "threadpool AddTask: " << GET_EVENT_TATALTIME(WorkQueue::AddTask)
            << "ns " << GET_EVENT_COUNT(WorkQueue::AddTask) << "times";
    uint64_t allocator_cost =
        GET_EVENT_TATALTIME(StreamSafeCUDAAllocator::Allocate) +
        GET_EVENT_TATALTIME(StreamSafeCUDAAllocator::Free);
    VLOG(1) << "StreamSafe Allocator: " << allocator_cost << "ns "
            << GET_EVENT_COUNT(StreamSafeCUDAAllocator::Allocate) +
                   GET_EVENT_COUNT(StreamSafeCUDAAllocator::Free)
            << "times";
    uint64_t autogrowth_allocator_cost =
        GET_EVENT_TATALTIME(AutoGrowthBestFitAllocator::Allocate) +
        GET_EVENT_TATALTIME(AutoGrowthBestFitAllocator::Free);
    VLOG(1) << "AutoGrowth Allocator: " << autogrowth_allocator_cost << "ns "
            << GET_EVENT_COUNT(AutoGrowthBestFitAllocator::Allocate) +
                   GET_EVENT_COUNT(AutoGrowthBestFitAllocator::Free)
            << "times";
    VLOG(1) << "kernel luanch: "
            << GET_EVENT_TATALTIME(Compute) - allocator_cost << "ns ";
    VLOG(1) << "op count: " << GET_EVENT_COUNT(Compute);
  } else {
    VLOG(1) << "static kernel=========";
    VLOG(1) << "data transform: " << GET_EVENT_TATALTIME(prepare_data) << "ns "
            << GET_EVENT_COUNT(prepare_data) << "times";
    VLOG(1) << "thread model==========";
    VLOG(1) << "threadpool AddTask: " << GET_EVENT_TATALTIME(WorkQueue::AddTask)
            << "ns " << GET_EVENT_COUNT(WorkQueue::AddTask) << "times";
    uint64_t allocator_cost =
        GET_EVENT_TATALTIME(AutoGrowthBestFitAllocator::Allocate) +
        GET_EVENT_TATALTIME(AutoGrowthBestFitAllocator::Free);
    VLOG(1) << "Allocator: " << allocator_cost << "ns "
            << GET_EVENT_COUNT(AutoGrowthBestFitAllocator::Allocate) +
                   GET_EVENT_COUNT(AutoGrowthBestFitAllocator::Free)
            << "times";
    VLOG(1) << "kernel luanch: "
            << GET_EVENT_TATALTIME(compute) - allocator_cost << "ns ";
    VLOG(1) << "op count: " << GET_EVENT_COUNT(compute);
  }
#undef GET_EVENT_TATALTIME
#undef GET_EVENT_COUNT
}

}  // namespace

void HostTracer::StartTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::READY || state_ == TracerState::STOPED, true,
      platform::errors::PreconditionNotMet("TracerState must be READY"));
  HostEventRecorder::GetInstance().GatherEvents();
  HostTraceLevel::GetInstance().SetLevel(trace_level_);
  state_ = TracerState::STARTED;
  VLOG(1) << "HostTracer::StartTracing level:" << trace_level_;
}

void HostTracer::StopTracing() {
  PADDLE_ENFORCE_EQ(
      state_, TracerState::STARTED,
      platform::errors::PreconditionNotMet("TracerState must be STARTED"));
  HostTraceLevel::GetInstance().SetLevel(HostTraceLevel::kDisabled);
  state_ = TracerState::STOPED;
  VLOG(1) << "HostTracer::StopTracing";
}

void HostTracer::CollectTraceData(TraceEventCollector* collector) {
  PADDLE_ENFORCE_EQ(
      state_, TracerState::STOPED,
      platform::errors::PreconditionNotMet("TracerState must be STOPED"));
  HostEventSection host_events =
      HostEventRecorder::GetInstance().GatherEvents();
  VLOG(1) << "HostTracer::CollectTraceData";
  ProcessHostEvents(host_events, collector);
  StatisticsHostEvents(*collector);
}

}  // namespace platform
}  // namespace paddle
