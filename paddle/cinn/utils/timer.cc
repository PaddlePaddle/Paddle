// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/utils/timer.h"

namespace cinn {
namespace utils {

float Timer::Stop() {
  end_ = std::chrono::high_resolution_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_);
  float ms = 1000. * static_cast<double>(ts.count()) *
             std::chrono::nanoseconds::period::num /
             std::chrono::nanoseconds::period::den;
  return ms;
}

void Timer::Start() { start_ = std::chrono::high_resolution_clock::now(); }

}  // namespace utils
}  // namespace cinn
