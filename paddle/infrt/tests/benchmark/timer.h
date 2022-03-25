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

#include <chrono>

namespace infrt {
namespace tests {

template <typename ClockT>
class ChronoTimer {
 public:
  using TimePoint = std::chrono::time_point<ClockT>;
  ChronoTimer() : start_{TimePoint::min()} {}
  void Clear() { start_ = TimePoint::min(); }
  void Start() { start_ = ClockT::now(); }

  uint32_t GetMs() {
    if (start_ != TimePoint::min()) {
      typename ClockT::duration diff = ClockT::now() - start_;
      return static_cast<uint32_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(diff).count());
    } else {
      return 0;
    }
  }

 private:
  TimePoint start_;
};

using WallClockTimer = ChronoTimer<std::chrono::steady_clock>;

class CpuClockTimer {
 public:
  CpuClockTimer() = default;
  void Clear() { start_ = 0; }
  void Start() { start_ = std::clock(); }
  uint32_t GetMs() {
    if (start_ != 0) {
      std::clock_t diff = std::clock() - start_;
      return static_cast<uint32_t>(diff * 1000 / CLOCKS_PER_SEC);
    } else {
      return 0;
    }
  }

 private:
  std::clock_t start_{0};
};

}  // namespace tests
}  // namespace infrt
