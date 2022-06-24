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

#include <set>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_python.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/platform/profiler/profiler.h"

TEST(ProfilerTest, TestHostTracer) {
  using paddle::platform::Profiler;
  using paddle::platform::ProfilerOptions;
  using paddle::platform::ProfilerResult;
  using paddle::platform::RecordInstantEvent;
  using paddle::platform::TracerEventType;
  ProfilerOptions options;
  options.trace_level = 2;
  options.trace_switch = 3;
  auto profiler = Profiler::Create(options);
  EXPECT_TRUE(profiler);
  profiler->Prepare();
  profiler->Start();
  {
    RecordInstantEvent(
        "TestTraceLevel_record1", TracerEventType::UserDefined, 2);
    RecordInstantEvent(
        "TestTraceLevel_record2", TracerEventType::UserDefined, 3);
  }
  auto profiler_result = profiler->Stop();
  auto nodetree = profiler_result->GetNodeTrees();
  std::set<std::string> host_events;
  for (const auto pair : nodetree->Traverse(true)) {
    for (const auto evt : pair.second) {
      host_events.insert(evt->Name());
    }
  }
  EXPECT_EQ(host_events.count("TestTraceLevel_record1"), 1u);
  EXPECT_EQ(host_events.count("TestTraceLevel_record2"), 0u);
}

TEST(ProfilerTest, TestCudaTracer) {
  using paddle::platform::Profiler;
  using paddle::platform::ProfilerOptions;
  using paddle::platform::ProfilerResult;
  ProfilerOptions options;
  options.trace_level = 0;
  options.trace_switch = 3;
  auto profiler = Profiler::Create(options);
  EXPECT_TRUE(profiler);
  profiler->Prepare();
  profiler->Start();
#ifdef PADDLE_WITH_CUDA
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaStreamSynchronize(stream);
#endif
#ifdef PADDLE_WITH_HIP
  hipStream_t stream;
  hipStreamCreate(&stream);
  hipStreamSynchronize(stream);
#endif
  auto profiler_result = profiler->Stop();
  auto nodetree = profiler_result->GetNodeTrees();
  std::vector<std::string> runtime_events;
  for (const auto pair : nodetree->Traverse(true)) {
    for (const auto host_node : pair.second) {
      for (auto runtime_node : host_node->GetRuntimeTraceEventNodes()) {
        runtime_events.push_back(runtime_node->Name());
      }
    }
  }
#ifdef PADDLE_WITH_CUPTI
  EXPECT_GT(runtime_events.size(), 0u);
#endif
}

TEST(ProfilerTest, TestHostTracerForMem) {
  using paddle::platform::CPUPlace;
  using paddle::platform::EnableHostEventRecorder;
  using paddle::platform::MemTraceEventNode;
  using paddle::platform::Profiler;
  using paddle::platform::ProfilerOptions;
  using paddle::platform::ProfilerResult;
  using paddle::platform::RecordEvent;
  using paddle::platform::RecordInstantEvent;
  using paddle::platform::RecordMemEvent;
  using paddle::platform::TracerEventType;
  using paddle::platform::TracerMemEventType;
  ProfilerOptions options;
  options.trace_level = 1;
  options.trace_switch = 3;
  auto profiler = Profiler::Create(options);
  EXPECT_TRUE(profiler);
  EnableHostEventRecorder();
  profiler->Prepare();
  profiler->Start();
  {
    RecordEvent event1(
        "TestTracerForMem_phase1", TracerEventType::UserDefined, 1);
    RecordMemEvent(reinterpret_cast<void*>(0),
                   CPUPlace(),
                   1024,
                   TracerMemEventType::Allocate);
    RecordMemEvent(
        reinterpret_cast<void*>(0), CPUPlace(), 1024, TracerMemEventType::Free);
  }
  {
    RecordEvent event2(
        "TestTracerForMem_phase2", TracerEventType::UserDefined, 1);
    RecordMemEvent(reinterpret_cast<void*>(1024),
                   CPUPlace(),
                   1024,
                   TracerMemEventType::Allocate);
    RecordMemEvent(reinterpret_cast<void*>(1024),
                   CPUPlace(),
                   1024,
                   TracerMemEventType::Free);
  }
  auto profiler_result = profiler->Stop();
  auto nodetree = profiler_result->GetNodeTrees();
}
