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

#include <stdio.h>

#include <cinttypes>
#include <cstdint>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <sys/times.h>
#include <unistd.h>
#endif

namespace paddle {
namespace platform {

class CpuUtilization {
 public:
  CpuUtilization() {}
  void RecordBeginTimeInfo();
  void RecordEndTimeInfo();
  float GetCpuUtilization();
  float GetCpuCurProcessUtilization();

 private:
#ifdef _MSC_VER
  FILETIME start_, end_;
  FILETIME process_user_time_start_, process_user_time_end_;
  FILETIME process_kernel_time_start_, process_kernel_time_end_;
  FILETIME system_user_time_start_, system_user_time_end_;
  FILETIME system_kernel_time_start_, system_kernel_time_end_;
  FILETIME system_idle_time_start_, system_idle_time_end_;
  FILETIME process_creation_time_, process_exit_time_;
#else
  clock_t start_, end_;
  uint64_t idle_start_, idle_end_;
  uint64_t iowait_start_, iowait_end_;
  uint64_t nice_time_start_, nice_time_end_;
  uint64_t irq_start_, irq_end_;
  uint64_t softirq_start_, softirq_end_;
  uint64_t steal_start_, steal_end_;
  struct tms system_tms_start_, system_tms_end_;
  struct tms process_tms_start_, process_tms_end_;
#endif
};

}  // namespace platform
}  // namespace paddle
