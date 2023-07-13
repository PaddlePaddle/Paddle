// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/utils/profiler.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(RecordEvent, HOST) {
  using cinn::utils::EventType;
  using cinn::utils::HostEventRecorder;
  using cinn::utils::ProfilerHelper;
  using cinn::utils::RecordEvent;

  ProfilerHelper::EnableCPU();

  LOG(INFO) << "Usage 1: RecordEvent for HOST";
  std::vector<EventType> types = {EventType::kOrdinary,
                                  EventType::kCompile,
                                  EventType::kCompile,
                                  EventType::kInstruction};
  for (int i = 0; i < 4; ++i) {
    std::string name = "evs_op_" + std::to_string(i);
    RecordEvent record_event(name, types[i]);
    int counter = 1;
    while (counter != i * 1000) counter++;
  }

  auto &events = HostEventRecorder::GetInstance().Events();
  EXPECT_EQ(events.size(), 4U);
  for (int i = 0; i < 4; ++i) {
    auto &event = events[i];
    std::string name = "evs_op_" + std::to_string(i);
    EXPECT_EQ(event.annotation_, name);
    EXPECT_GT(event.duration_, 0.0);
    EXPECT_EQ(event.type_, types[i]);
    LOG(INFO) << name << " cost :" << event.duration_ << " ms.";
  }

  LOG(INFO) << HostEventRecorder::Table();
  /*
    40: ------------------------->     Profiling Report
    <------------------------- 40: 40:  Category             Name CostTime(ms)
    Ratio in Category(%)  Ratio in Total(%) 40: 40:  Ordinary evs_op_0
    9725.647664          100.000000           99.999827 40:  Instruction
    evs_op_3             0.006967             100.000000           0.000072 40:
    Compile              evs_op_1             0.005083             51.536044
    0.000052 40:  Compile              evs_op_2 0.004780             48.463956
    0.000049
  */

  LOG(INFO) << "Usage 2: Nested RecordEvent for HOST";
  HostEventRecorder::GetInstance().Clear();
  EXPECT_EQ(events.size(), 0U);

  for (int i = 0; i < 4; ++i) {
    std::string name = "ano_evs_op_" + std::to_string(i);
    RecordEvent record_event(name);
    int counter = 0;
    while (counter != i * 10) counter++;
    {
      std::string nested_name = "nested_ano_evs_op_" + std::to_string(i);
      RecordEvent nested_record_event(nested_name);
      int nested_counter = 1;
      while (nested_counter != i * 100) nested_counter++;
    }
  }
  EXPECT_EQ(events.size(), 8U);
}
