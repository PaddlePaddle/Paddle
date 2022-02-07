/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

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
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/platform/profiler/profiler.h"

TEST(ProfilerTest, TestHostTracer) {
  using paddle::platform::ProfilerOptions;
  using paddle::platform::Profiler;
  using paddle::platform::RecordInstantEvent;
  using paddle::platform::TracerEventType;
  ProfilerOptions options;
  options.trace_level = 2;
  auto profiler = Profiler::Create(options);
  EXPECT_TRUE(profiler);
  profiler->Prepare();
  profiler->Start();
  {
    RecordInstantEvent("TestTraceLevel_record1", TracerEventType::UserDefined,
                       2);
    RecordInstantEvent("TestTraceLevel_record2", TracerEventType::UserDefined,
                       3);
  }
  auto collector = profiler->Stop();
  std::set<std::string> host_events;
  for (const auto evt : collector.HostEvents()) {
    host_events.insert(evt.name);
  }
  EXPECT_EQ(host_events.count("TestTraceLevel_record1"), 1u);
  EXPECT_EQ(host_events.count("TestTraceLevel_record2"), 0u);
}
