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

#pragma once
#include "paddle/fluid/platform/profiler/profiler.h"
// #include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/platform/profiler/trace_event.h"
namespace phi {

class KernelProfiler {
 public:
  KernelProfiler() {
    paddle::platform::ProfilerOptions options;
    options.trace_level = 1;
    options.trace_switch = 3;
    profiler_ = paddle::platform::Profiler::Create(options);
    profiler_->Prepare();
  }

  void Start() { profiler_->Start(); }

  void Stop() { result_ = profiler_->Stop(); }

  paddle::platform::RecordEvent&& RecordEvent(const std::string& name) {
    return std::move(paddle::platform::RecordEvent(
        name, paddle::platform::TracerEventType::OperatorInner, 0));
  }

  void GetPerfResults() {
    VLOG(3) << "========== Perf Results ==========";
    auto& nodetree = result_->GetNodeTrees();
    std::vector<std::string> runtime_events;
    for (const auto pair : nodetree->Traverse(true)) {
      for (const auto host_node : pair.second) {
        VLOG(3) << "host_node: " << host_node->Name();
        for (auto runtime_node : host_node->GetRuntimeTraceEventNodes()) {
          VLOG(3) << "kernel name: " << runtime_node->Name();
          runtime_events.push_back(runtime_node->Name());
        }
      }
    }
  }

  ~KernelProfiler() {}

 private:
  std::unique_ptr<paddle::platform::Profiler::Profiler> profiler_ = nullptr;
  std::unique_ptr<paddle::platform::ProfilerResult> result_;
  uint64_t elapsed_time_;
};

}  // namespace phi
