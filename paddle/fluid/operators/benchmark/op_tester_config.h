/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <istream>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <sys/time.h>
#endif
#include <chrono>  // NOLINT

namespace paddle {
namespace operators {
namespace benchmark {

struct Timer {
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

struct OpInputConfig {
  explicit OpInputConfig(std::istream& is);

  std::string name;
  std::vector<int64_t> dims;
};

struct OpTesterConfig {
  OpTesterConfig() {}
  explicit OpTesterConfig(const std::string& filename);
  void Init(std::istream& is);

  const OpInputConfig* GetInput(const std::string& name);

  std::string op_type;
  std::vector<OpInputConfig> inputs;
  int use_gpu{0};
  int repeat{1};
  int profile{0};
  int print_debug_string{0};
  double runtime{0.0};
};

}  // namespace benchmark
}  // namespace operators
}  // namespace paddle
