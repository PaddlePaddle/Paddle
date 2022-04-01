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

#include "paddle/fluid/platform/profiler/profiler.h"
#include "glog/logging.h"
#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#endif
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler/cuda_tracer.h"
#include "paddle/fluid/platform/profiler/extra_info.h"
#include "paddle/fluid/platform/profiler/host_tracer.h"
#include "paddle/fluid/platform/profiler/trace_event_collector.h"
#include "paddle/fluid/platform/profiler/utils.h"

namespace paddle {
namespace platform {

void SynchronizeAllDevice();

std::atomic<bool> Profiler::alive_{false};

std::unique_ptr<Profiler> Profiler::Create(const ProfilerOptions& options) {
  if (alive_.exchange(true)) {
    return nullptr;
  }
  return std::unique_ptr<Profiler>(new Profiler(options));
}

bool Profiler::IsCuptiSupported() {
  bool supported = false;
#ifdef PADDLE_WITH_CUPTI
  supported = true;
#endif
  return supported;
}

Profiler::Profiler(const ProfilerOptions& options) {
  options_ = options;
  std::bitset<32> trace_switch(options_.trace_switch);
  if (trace_switch.test(kProfileCPUOptionBit)) {
    HostTracerOptions host_tracer_options;
    host_tracer_options.trace_level = options_.trace_level;
    tracers_.emplace_back(new HostTracer(host_tracer_options), true);
  }
  if (trace_switch.test(kProfileGPUOptionBit)) {
    tracers_.emplace_back(&CudaTracer::GetInstance(), false);
  }
}

Profiler::~Profiler() { alive_.store(false); }

void Profiler::Prepare() {
  for (auto& tracer : tracers_) {
    tracer.Get().PrepareTracing();
  }
}

void Profiler::Start() {
  SynchronizeAllDevice();
  for (auto& tracer : tracers_) {
    tracer.Get().StartTracing();
  }
  cpu_utilization_.RecordBeginTimeInfo();
}

std::unique_ptr<ProfilerResult> Profiler::Stop() {
  SynchronizeAllDevice();
  TraceEventCollector collector;
  for (auto& tracer : tracers_) {
    tracer.Get().StopTracing();
    tracer.Get().CollectTraceData(&collector);
  }
  std::unique_ptr<NodeTrees> tree(new NodeTrees(collector.HostEvents(),
                                                collector.RuntimeEvents(),
                                                collector.DeviceEvents()));
  cpu_utilization_.RecordEndTimeInfo();
  ExtraInfo extrainfo;
  extrainfo.AddExtraInfo(std::string("System Cpu Utilization"),
                         std::string("%f"),
                         cpu_utilization_.GetCpuUtilization());
  extrainfo.AddExtraInfo(std::string("Process Cpu Utilization"),
                         std::string("%f"),
                         cpu_utilization_.GetCpuCurProcessUtilization());
  const std::unordered_map<uint64_t, std::string> thread_names =
      collector.ThreadNames();
  for (const auto& kv : thread_names) {
    extrainfo.AddExtraInfo(string_format(std::string("%llu"), kv.first),
                           std::string("%s"), kv.second.c_str());
  }
  return std::unique_ptr<ProfilerResult>(
      new platform::ProfilerResult(std::move(tree), extrainfo));
}

}  // namespace platform
}  // namespace paddle
