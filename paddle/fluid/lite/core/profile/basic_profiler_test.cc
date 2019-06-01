// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/profile/basic_profiler.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <chrono>  // NOLINT
#include <thread>  // NOLINT

namespace paddle {
namespace lite {
namespace profile {

TEST(basic_record, init) {
  BasicTimer timer;
  timer.SetKey("hello");
}

TEST(basic_profile, init) {
  auto& rcd = BasicProfiler<BasicTimer>::Global().NewRcd("fc");
  for (int i = 11; i < 100; i++) {
    rcd.Log(i);
  }

  LOG(INFO) << BasicProfiler<BasicTimer>::Global().basic_repr();
}

TEST(basic_profile, real_latency) {
  LITE_PROFILE_ONE(test0);
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));
}

}  // namespace profile
}  // namespace lite
}  // namespace paddle
