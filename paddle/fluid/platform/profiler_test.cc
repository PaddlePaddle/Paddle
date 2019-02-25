/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/profiler.h"
#include <string>
#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif
#include "gtest/gtest.h"

TEST(Event, CpuElapsedTime) {
  using paddle::platform::Event;
  using paddle::platform::EventType;

  Event start_event(EventType::kPushRange, "test", 0);
  int counter = 0;
  while (counter != 1000) {
    counter++;
  }
  Event stop_event(EventType::kPopRange, "test", 0);
  EXPECT_GT(start_event.CpuElapsedMs(stop_event), 0);
}

TEST(RecordEvent, RecordEvent) {
  using paddle::platform::Event;
  using paddle::platform::EventType;
  using paddle::platform::RecordEvent;
  using paddle::platform::PushEvent;
  using paddle::platform::PopEvent;
  using paddle::platform::ProfilerState;
  using paddle::platform::EventSortingKey;

  ProfilerState state = ProfilerState::kCPU;
  EnableProfiler(state);

  /* Usage 1:
  *  PushEvent(evt_name);
  *  ...
  *  code to be analyzed
  *  ...
  * PopEvent(evt_name);
  */
  LOG(INFO) << "Usage 1: PushEvent & PopEvent";
  for (int loop = 0; loop < 3; ++loop) {
    for (int i = 1; i < 5; ++i) {
      std::string name = "op_" + std::to_string(i);
      PushEvent(name);
      int counter = 1;
      while (counter != i * 1000) counter++;
      PopEvent(name);
    }
  }

  /* Usage 2:
   * {
   *   RecordEvent record_event(name);
   *   ...
   *   code to be analyzed
   *   ...
   * }
   */
  LOG(INFO) << "Usage 2: RecordEvent";
  for (int i = 1; i < 5; ++i) {
    std::string name = "evs_op_" + std::to_string(i);
    RecordEvent record_event(name);
    int counter = 1;
    while (counter != i * 1000) counter++;
  }

  /* Usage 3
   * {
   *   RecordEvent record_event(name1, dev_ctx);
   *   ...
   *   code to be analyzed
   *   ...
   *   {
   *     RecordEvent nested_record_event(name2, dev_ctx);
   *     ...
   *     code to be analyzed
   *     ...
   *   }
   * }
   */
  LOG(INFO) << "Usage 3: nested RecordEvent";
  for (int i = 1; i < 5; ++i) {
    std::string name = "ano_evs_op_" + std::to_string(i);
    RecordEvent record_event(name);
    int counter = 1;
    while (counter != i * 100) counter++;
    {
      std::string nested_name = "nested_ano_evs_op_" + std::to_string(i);
      RecordEvent nested_record_event(nested_name);
      int nested_counter = 1;
      while (nested_counter != i * 100) nested_counter++;
    }
  }

  // Bad Usage:
  PushEvent("event_without_pop");
  PopEvent("event_without_push");
  std::vector<std::vector<Event>> events = paddle::platform::GetAllEvents();

  int cuda_startup_count = 0;
  int start_profiler_count = 0;
  for (size_t i = 0; i < events.size(); ++i) {
    for (size_t j = 0; j < events[i].size(); ++j) {
      if (events[i][j].name() == "_cuda_startup_") ++cuda_startup_count;
      if (events[i][j].name() == "_start_profiler_") ++start_profiler_count;
      if (events[i][j].name() == "push") {
        EXPECT_EQ(events[i][j + 1].name(), "pop");
#ifdef PADDLE_WITH_CUDA
        EXPECT_GT(events[i][j].CudaElapsedMs(events[i][j + 1]), 0);
#else
        EXPECT_GT(events[i][j].CpuElapsedMs(events[i][j + 1]), 0);
#endif
      }
    }
  }
  EXPECT_EQ(cuda_startup_count % 5, 0);
  EXPECT_EQ(start_profiler_count, 1);

  // Will remove parsing-related code from test later
  DisableProfiler(EventSortingKey::kTotal, "/tmp/profiler");
}

#ifdef PADDLE_WITH_CUDA
TEST(TMP, stream_wait) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaStreamSynchronize(stream);
  cudaStreamSynchronize(stream);
  cudaStreamSynchronize(stream);
}
#endif
