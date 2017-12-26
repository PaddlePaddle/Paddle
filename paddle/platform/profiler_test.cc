/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/platform/profiler.h"
#include "gtest/gtest.h"

TEST(Event, CpuElapsedTime) {
  using paddle::platform::Event;
  using paddle::platform::EventKind;

  Event start_event(EventKind::kPushRange, "test", 0);
  EXPECT_TRUE(start_event.has_cuda() == false);
  int counter = 0;
  while (counter != 1000) {
    counter++;
  }
  Event stop_event(EventKind::kPopRange, "test", 0);
  EXPECT_GT(start_event.CpuElapsedUs(stop_event), 0);
}

#ifdef PADDLE_WITH_CUDA
TEST(Event, CudaElapsedTime) {
  using paddle::platform::DeviceContext;
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::GPUPlace;
  using paddle::platform::Event;
  using paddle::platform::EventKind;

  DeviceContext* dev_ctx = new CUDADeviceContext(GPUPlace(0));
  Event start_event(EventKind::kPushRange, "test", 0, dev_ctx);
  EXPECT_TRUE(start_event.has_cuda() == true);
  int counter = 0;
  while (counter != 1000) {
    counter++;
  }
  Event stop_event(EventKind::kPopRange, "test", 0, dev_ctx);
  EXPECT_GT(start_event.CudaElapsedUs(stop_event), 0);
}
#endif

TEST(RecordEvent, RecordEvent) {
  using paddle::platform::DeviceContext;
  using paddle::platform::Event;
  using paddle::platform::EventKind;
  using paddle::platform::RecordEvent;
  using paddle::platform::ProfilerState;

  ProfilerState state = ProfilerState::kCPU;
  DeviceContext* dev_ctx = nullptr;
#ifdef PADDLE_WITH_CUDA
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::GPUPlace;
  state = ProfilerState::kCUDA;
  dev_ctx =
      new paddle::platform::CUDADeviceContext(paddle::platform::GPUPlace(0));
#endif
  EnableProfiler(state);

  /* Usage 1:
  *  PushEvent(evt_name, dev_ctx);
  *  ...
  *  code to time
  *  ...
  * PopEvent(evt_name, dev_ctx);
  */
  for (int i = 1; i < 5; ++i) {
    std::string name = "op_" + std::to_string(i);
    PushEvent(name, dev_ctx);
    int counter = 1;
    while (counter != i * 1000) counter++;
    PopEvent(name, dev_ctx);
  }

  /* Usage 2:
   * {
   *   RecordEvent record_event(name, dev_ctx);
   *   ...
   * }
   */
  for (int i = 1; i < 5; ++i) {
    std::string name = "evs_op_" + std::to_string(i);
    RecordEvent record_event(name, dev_ctx);
    int counter = 1;
    while (counter != i * 1000) counter++;
  }
  std::vector<std::vector<Event>> events = paddle::platform::DisableProfiler();
  int cuda_startup_count = 0;
  int start_profiler_count = 0;
  int stop_profiler_count = 0;
  ParseEvents(events);
  for (size_t i = 0; i < events.size(); ++i) {
    for (size_t j = 0; j < events[i].size(); ++j) {
      if (events[i][j].name() == "_cuda_startup_") ++cuda_startup_count;
      if (events[i][j].name() == "_start_profiler_") ++start_profiler_count;
      if (events[i][j].name() == "_stop_profiler_") ++stop_profiler_count;
      if (events[i][j].name() == "push") {
        EXPECT_EQ(events[i][j + 1].name(), "pop");
#ifdef PADDLE_WITH_CUDA
        EXPECT_GT(events[i][j].CudaElapsedUs(events[i][j + 1]), 0);
#else
        EXPECT_GT(events[i][j].CpuElapsedUs(events[i][j + 1]), 0);
#endif
      }
    }
  }
  EXPECT_EQ(cuda_startup_count % 5, 0);
  EXPECT_EQ(start_profiler_count, 1);
  EXPECT_EQ(stop_profiler_count, 1);
}
