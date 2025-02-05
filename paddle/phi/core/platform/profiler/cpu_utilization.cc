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

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include "paddle/phi/core/platform/profiler/cpu_utilization.h"
#include <array>
#include "glog/logging.h"

namespace paddle::platform {

#ifdef _MSC_VER
static uint64_t FileTimeToUint64(FILETIME time) {
  uint64_t low_part = time.dwLowDateTime;
  uint64_t high_part = time.dwHighDateTime;
  uint64_t result = (high_part << 32) | low_part;
  return result;
}
#endif

void CpuUtilization::RecordBeginTimeInfo() {
#if defined(_MSC_VER)
  HANDLE process_handle = GetCurrentProcess();
  GetSystemTimeAsFileTime(&start_);
  GetSystemTimes(&system_idle_time_start_,
                 &system_kernel_time_start_,
                 &system_user_time_start_);
  GetProcessTimes(process_handle,
                  &process_creation_time_,
                  &process_exit_time_,
                  &process_kernel_time_start_,
                  &process_user_time_start_);

#elif defined(__linux__)
  start_ = times(&process_tms_start_);
#define proc_path_size 1024
  static char proc_stat_path[proc_path_size] = "/proc/stat";  // NOLINT
  FILE *stat_file = fopen(proc_stat_path, "r");
  if (stat_file != nullptr) {
    std::array<char, 200> temp_str = {};
    uint64_t temp_lu;
    int retval =
        fscanf(stat_file,
               "%s %" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64
               "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64,
               temp_str.data(),
               &system_tms_start_.tms_utime,
               &nice_time_start_,
               &system_tms_start_.tms_stime,
               &idle_start_,
               &iowait_start_,
               &irq_start_,
               &softirq_start_,
               &steal_start_,
               &temp_lu,
               &temp_lu);
    if (retval != 11) {
      LOG(WARNING)
          << "Failed to read cpu utilization information at record beginning."
          << std::endl;
    }
    fclose(stat_file);
  }
#else
#endif
}

void CpuUtilization::RecordEndTimeInfo() {
#if defined(_MSC_VER)
  HANDLE process_handle = GetCurrentProcess();
  GetSystemTimeAsFileTime(&end_);
  GetSystemTimes(
      &system_idle_time_end_, &system_kernel_time_end_, &system_user_time_end_);
  GetProcessTimes(process_handle,
                  &process_creation_time_,
                  &process_exit_time_,
                  &process_kernel_time_end_,
                  &process_user_time_end_);
#elif defined(__linux__)
  end_ = times(&process_tms_end_);
#define proc_path_size 1024
  static char proc_stat_path[proc_path_size] = "/proc/stat";  // NOLINT
  FILE *stat_file = fopen(proc_stat_path, "r");
  if (stat_file != nullptr) {
    std::array<char, 200> temp_str = {};
    uint64_t temp_lu;
    int retval =
        fscanf(stat_file,
               "%s %" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64
               "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64,
               temp_str.data(),
               &system_tms_end_.tms_utime,
               &nice_time_end_,
               &system_tms_end_.tms_stime,
               &idle_end_,
               &iowait_end_,
               &irq_end_,
               &softirq_end_,
               &steal_end_,
               &temp_lu,
               &temp_lu);

    if (retval != 11) {
      LOG(WARNING)
          << "Failed to read cpu utilization information at record end."
          << std::endl;
    }
    fclose(stat_file);
  }
#else
#endif
}

float CpuUtilization::GetCpuUtilization() {
  float cpu_utilization = 0.0;
#if defined(_MSC_VER)
  uint64_t system_user_time_start = FileTimeToUint64(system_user_time_start_);
  uint64_t system_user_time_end = FileTimeToUint64(system_user_time_end_);
  uint64_t system_kernel_time_start =
      FileTimeToUint64(system_kernel_time_start_);
  uint64_t system_kernel_time_end = FileTimeToUint64(system_kernel_time_end_);
  uint64_t system_idle_time_start = FileTimeToUint64(system_idle_time_start_);
  uint64_t system_idle_time_end = FileTimeToUint64(system_idle_time_end_);
  float busy_time = (system_kernel_time_end - system_kernel_time_start) +
                    (system_user_time_end - system_user_time_start);
  float idle_time = system_idle_time_end - system_idle_time_start;
  if (busy_time + idle_time != 0) {
    cpu_utilization = busy_time / (busy_time + idle_time);
  }
#elif defined(__linux__)
  float busy_time = (system_tms_end_.tms_utime - system_tms_start_.tms_utime) +
                    (system_tms_end_.tms_stime - system_tms_start_.tms_stime) +
                    (nice_time_end_ - nice_time_start_) +
                    (irq_end_ - irq_start_) + (softirq_end_ - softirq_start_) +
                    (steal_end_ - steal_start_);  // NOLINT
  float idle_time = (idle_end_ - idle_start_) + (iowait_end_ - iowait_start_);
  if (busy_time + idle_time != 0) {
    cpu_utilization = busy_time / (busy_time + idle_time);
  }
#else
  LOG(WARNING)
      << "Current System is not supported to get system cpu utilization"
      << cpu_utilization << std::endl;
#endif
  return cpu_utilization;
}

float CpuUtilization::GetCpuCurProcessUtilization() {
  float cpu_process_utilization = 0.0;
#ifdef _MSC_VER
  uint64_t process_user_time_start = FileTimeToUint64(process_user_time_start_);
  uint64_t process_user_time_end = FileTimeToUint64(process_user_time_end_);
  uint64_t process_kernel_time_start =
      FileTimeToUint64(process_kernel_time_start_);
  uint64_t process_kernel_time_end = FileTimeToUint64(process_kernel_time_end_);
  uint64_t start = FileTimeToUint64(start_);
  uint64_t end = FileTimeToUint64(end_);
  float busy_time = (process_kernel_time_end - process_kernel_time_start) +
                    (process_user_time_end - process_user_time_start);
  if (end - start != 0) {
    cpu_process_utilization = busy_time / (end - start);
  }
#elif defined(__linux__)
  float busy_time =
      (process_tms_end_.tms_utime - process_tms_start_.tms_utime) +
      (process_tms_end_.tms_stime - process_tms_start_.tms_stime);  // NOLINT
  if (end_ - start_ != 0) {
    cpu_process_utilization = busy_time / (end_ - start_);
  }
#else
  LOG(WARNING)
      << "Current System is not supported to get process cpu utilization"
      << cpu_process_utilization << std::endl;
#endif
  return cpu_process_utilization;
}

}  // namespace paddle::platform
