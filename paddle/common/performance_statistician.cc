// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/common/performance_statistician.h"

#include <fstream>
#include <numeric>
#include <stack>
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"

namespace common {

std::vector<TimeDuration> PerformanceReporter::ExtractDuration(
    const std::vector<TimePointInfo>& records, bool contain_recursive) {
  std::vector<TimeDuration> durations;
  std::stack<TimePointInfo> stk;
  for (const auto& info : records) {
    if (info.is_start) {
      stk.push(info);
    } else {
      PADDLE_ENFORCE_EQ(
          (!stk.empty() && stk.top().is_start),
          true,
          common::errors::InvalidArgument(
              "There is a problem with the call stack of records"));
      auto start = stk.top();
      stk.pop();
      if (contain_recursive || stk.empty()) {
        durations.emplace_back(info.time_point - start.time_point);
      }
    }
  }
  return durations;
}

TimeDuration PerformanceReporter::Sum(
    const std::vector<TimeDuration>& records) {
  return std::accumulate(records.begin(), records.end(), TimeDuration::zero());
}

TimeDuration PerformanceReporter::Mean(
    const std::vector<TimeDuration>& records) {
  if (records.empty()) return TimeDuration::zero();
  return Sum(records) / records.size();
}

TimeDuration PerformanceReporter::Max(
    const std::vector<TimeDuration>& records) {
  return *std::max_element(records.begin(), records.end());
}

TimeDuration PerformanceReporter::Min(
    const std::vector<TimeDuration>& records) {
  return *std::min_element(records.begin(), records.end());
}

std::vector<TimeDuration> PerformanceReporter::TopK(
    const std::vector<TimeDuration>& records, int top_count) {
  std::vector<TimeDuration> top_k(top_count);
  std::partial_sort_copy(records.begin(),
                         records.end(),
                         top_k.begin(),
                         top_k.end(),
                         std::greater<TimeDuration>());
  return top_k;
}

TimeDuration PerformanceReporter::TrimMean(
    const std::vector<TimeDuration>& durations) {
  int top_count = durations.size();
  if (top_count == 0) return TimeDuration::zero();
  auto top_k = TopK(durations, top_count);
  int remove_num = top_count / 5;
  auto avg_time = std::accumulate(top_k.begin() + remove_num,
                                  top_k.end() - remove_num,
                                  TimeDuration::zero());
  return avg_time / (top_count - 2 * remove_num);
}

std::string PerformanceReporter::Report(
    const std::vector<TimePointInfo>& records) {
  if (records.empty()) return "[No Record]";
  std::stringstream ss;
  std::string unit = "us";
  auto durations = ExtractDuration(records);
  int top_count = durations.size();
  auto total_time = std::chrono::duration_cast<TimeDuration>(Sum(durations));
  auto mean_time = std::chrono::duration_cast<TimeDuration>(Mean(durations));
  auto trim_mean_time =
      std::chrono::duration_cast<TimeDuration>(TrimMean(durations));
  auto max_time = std::chrono::duration_cast<TimeDuration>(Max(durations));
  auto min_time = std::chrono::duration_cast<TimeDuration>(Min(durations));
  auto top_k = TopK(durations, top_count);
  ss << "Call Count = " << durations.size()
     << "\t Total Time = " << total_time.count() << unit
     << "\t Mean Time = " << mean_time.count() << unit
     << "\t TrimMean Time = " << trim_mean_time.count() << unit
     << "\t Max Time = " << max_time.count() << unit
     << "\t Min Time = " << min_time.count() << unit << "\n";

  ss << "Records " << top_count << ": [";
  for (size_t i = 0; i < durations.size(); ++i) {
    ss << i + 1 << ": "
       << std::chrono::duration_cast<TimeDuration>(durations[i]).count() << unit
       << "  ";
  }
  ss << "]\n";
  ss << "Top " << top_count << ": [";
  for (size_t i = 0; i < top_k.size(); ++i) {
    ss << i + 1 << ": "
       << std::chrono::duration_cast<TimeDuration>(top_k[i]).count() << unit
       << "  ";
  }
  ss << "]\n";

  return ss.str();
}

std::string PerformanceReporter::Report(const PerformanceStatistician& stat) {
  std::stringstream ss;
  ss << "\n";
  ss << "=============================================================Start "
        "Report=============================================================\n";
  std::vector<std::string> labels = stat.Labels();
  for (const std::string& label : labels) {
    // std::cerr << "label: " << label << std::endl;
    ss << "Label = [" << label << "]\n";
    ss << Report(stat.Record(label));
    // std::cerr << "rep: " << ss.str() << std::endl;
    ss << "--------------------------------------------------------------------"
          "----------------------------------------------------------------\n";
  }
  ss << "==============================================================End "
        "Report=============================================================="
        "\n";
  return ss.str();
}

void PerformanceReporter::WriteToFile(const std::string& file,
                                      const std::string& report) {
  std::ofstream ofs;
  ofs.open(file, std::ofstream::out | std::ofstream::ate);
  if (ofs.is_open()) {
    ofs << report << std::flush;
  }
}

void PerformanceStatisticsStart(const std::string& label) {
  common::PerformanceStatistician& ps =
      common::PerformanceStatistician::Instance();
  ps.Start(label);
}

void PerformanceStatisticsEnd(const std::string& label) {
  common::PerformanceStatistician& ps =
      common::PerformanceStatistician::Instance();
  ps.End(label);
}

}  // namespace common
