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

#include <time.h>

#include "gtest/gtest.h"
#include "paddle/fluid/platform/profiler/cpu_utilization.h"

using paddle::platform::CpuUtilization;

TEST(CpuUtilizationTest, case0) {
  CpuUtilization instance;
#ifdef _MSC_VER
#else
  instance.RecordBeginTimeInfo();
  sleep(3);
  instance.RecordEndTimeInfo();
  instance.GetCpuUtilization();
  instance.GetCpuCurProcessUtilization();
  instance.start_ = 0;
  instance.end_ = 100;
  instance.idle_start_ = 50;
  instance.idle_end_ = 150;
  instance.system_tms_start_.tms_utime = 10;
  instance.system_tms_end_.tms_utime = 30;
  instance.process_tms_start_.tms_utime = 20;
  instance.process_tms_end_.tms_utime = 25;
  instance.system_tms_start_.tms_stime = 40;
  instance.system_tms_end_.tms_stime = 45;
  instance.process_tms_start_.tms_stime = 42;
  instance.process_tms_end_.tms_stime = 43;
  ASSERT_FLOAT_EQ(instance.GetCpuUtilization(), 0.2);
  ASSERT_FLOAT_EQ(instance.GetCpuCurProcessUtilization(), 0.06);
#endif
}
