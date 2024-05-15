/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <chrono>  // NOLINT [build/c++11]
#include <string>
#include "paddle/fluid/platform/device/gcu/utils/types.h"

namespace paddle {
namespace platform {
namespace gcu {

struct Recorder {
  explicit Recorder(const char *const profiler_env_value) {
    if (profiler_env_value != nullptr) {
      enable_flag = (std::string(profiler_env_value) == "true");
    }
  }

  ~Recorder() = default;

  bool IsEnableProfiler() { return enable_flag; }

  double Cost(const std::chrono::time_point<std::chrono::system_clock> &start,
              const std::chrono::time_point<std::chrono::system_clock> &end) {
    if (!IsEnableProfiler()) {
      return 0.0;
    }
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto run_time = static_cast<double>(duration.count()) *
                    std::chrono::microseconds::period::num /
                    std::chrono::microseconds::period::den;
    return run_time * 1000;
  }

 public:
  double time_total = 0;
  double time_init = 0;
  double time_update_input_memory = 0;
  double time_update_output_memory = 0;
  double time_update_memory = 0;
  double time_weights_h2d = 0;
  double time_weights_d2h = 0;
  double time_weights_inv_trans = 0;
  double time_learning_rate_h2d = 0;
  double time_dist_brd_weights = 0;
  double time_executable_run = 0;
  double time_dist_allreduce = 0;
  double time_weights_post_process = 0;
  double time_result_d2h = 0;
  double time_sync = 0;

 private:
  bool enable_flag = false;
};
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
