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

#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>

#include "paddle/infrt/tests/benchmark/timer.h"

#include "llvm/Support/raw_ostream.h"
#include "paddle/infrt/api/infrt_api.h"
#include "paddle/infrt/backends/host/phi_allocator.h"
#include "paddle/infrt/common/buffer.h"
#include "paddle/infrt/common/dtype.h"

namespace infrt {
namespace tests {

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
      assert(p >= 0 && p < 1);
      return stats[stats.size() * p];
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

int main() {
  using namespace infrt;  // NOLINT

  InfRtConfig config;

  config.set_model_dir(
      "/shixiaowei02/Paddle-InfRT/Paddle/linear/linear.pdmodel");
  config.set_param_dir(
      "/shixiaowei02/Paddle-InfRT/Paddle/linear/linear.pdiparams");
  config.set_mlir_path(
      "/shixiaowei02/Paddle-InfRT/Paddle/paddle/infrt/tests/dialect/phi/"
      "linear_cpu.mlir");

  std::unique_ptr<InfRtPredictor> predictor = CreateInfRtPredictor(config);

  ::infrt::backends::CpuPhiAllocator cpu_allocator;
  ::phi::DenseTensor* input = predictor->GetInput(0);
  input->Resize({16, 784});
  input->AllocateFrom(&cpu_allocator, ::phi::DataType::FLOAT32);
  auto* input_data = reinterpret_cast<float*>(input->data());
  for (int i = 0; i < input->numel(); i++) input_data[i] = 1.0;

  tests::BenchmarkStats stats;

  for (size_t i = 0; i < 100; ++i) {
    stats.Start();
    for (size_t i = 0; i < 100; ++i) {
      predictor->Run();
    }
    stats.Stop();
  }

  std::cout << stats.Summerize({0, 0.25, 0.5, 0.75, 0.99});

  return 0;
}
