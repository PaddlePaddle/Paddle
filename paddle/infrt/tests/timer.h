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

<<<<<<< HEAD
#include <algorithm>
#include <chrono>
#include <ctime>
#include <sstream>
#include <vector>
=======
#include <chrono>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace infrt {
namespace tests {

template <typename ClockT>
class ChronoTimer {
 public:
  using TimePoint = std::chrono::time_point<ClockT>;
  ChronoTimer() : start_{TimePoint::min()} {}
  void Clear() { start_ = TimePoint::min(); }
  void Start() { start_ = ClockT::now(); }

  double GetMs() {
    auto diff = ClockT::now() - start_;
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::duration<double>>(diff)
                   .count()) *
           1000.0;
  }

 private:
  TimePoint start_;
};

<<<<<<< HEAD
// To learn more about the difference between system_clock and steady_clock,
// please refer to https://www.cnblogs.com/zhongpan/p/7490657.html.
// To learn more about the difference between Wall Time and CPU Time,
// please refer to https://blog.csdn.net/aganlengzi/article/details/21888351
// and https://blog.csdn.net/filyouzicha/article/details/52447887.
using WallClockTimer = ChronoTimer<std::chrono::system_clock>;
=======
using WallClockTimer = ChronoTimer<std::chrono::steady_clock>;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

class CpuClockTimer {
 public:
  CpuClockTimer() = default;
  void Clear() { start_ = 0; }
  void Start() { start_ = std::clock(); }
  double GetMs() {
    std::clock_t diff = std::clock() - start_;
    return static_cast<double>(diff * 1000.0 / CLOCKS_PER_SEC);
  }

 private:
  std::clock_t start_{0};
};

class BenchmarkStats {
 public:
  void Start() {
    wall_timer_.Start();
    cpu_timer_.Start();
  }

  void Stop() {
    wall_time_.push_back(wall_timer_.GetMs());
    cpu_time_.push_back(cpu_timer_.GetMs());
  }

  std::string Summerize(const std::vector<float>& percents) {
    std::stringstream ss;
    std::sort(wall_time_.begin(), wall_time_.end());
    std::sort(cpu_time_.begin(), cpu_time_.end());
    auto percentile = [](float p, const std::vector<float>& stats) {
<<<<<<< HEAD
      size_t mark = stats.size() * p;
      mark = std::max(mark, static_cast<size_t>(0));
      mark = std::min(mark, stats.size() - 1);
      return stats[mark];
=======
      assert(p >= 0 && p < 1);
      return stats[stats.size() * p];
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    };
    for (auto p : percents) {
      ss << "=== Wall Time (ms): \n";
      ss << "  * percent " << std::to_string(static_cast<int>(p * 100));
      ss << ": " << percentile(p, wall_time_) << '\n';
    }
    for (auto p : percents) {
      ss << "=== CPU Time (ms): \n";
      ss << "  * percent " << std::to_string(static_cast<int>(p * 100));
      ss << ": " << percentile(p, cpu_time_) << '\n';
    }
    return ss.str();
  }

 private:
  WallClockTimer wall_timer_;
  std::vector<float> wall_time_;
  CpuClockTimer cpu_timer_;
  std::vector<float> cpu_time_;
};

}  // namespace tests
}  // namespace infrt
