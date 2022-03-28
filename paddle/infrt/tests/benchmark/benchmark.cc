// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>

#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/infrt/utils/timer.h"
#include "paddle_api.h"  // NOLINT

class Timer {
 public:
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

void benchmark(size_t layers, size_t num) {
  const std::string tag =
      "l" + std::to_string(layers) + "n" + std::to_string(num);
  const std::string model_name = tag + ".pdmodel";
  const std::string param_name = tag + ".pdiparams";

  const std::string model{"/shixiaowei02/Paddle-InfRT/Paddle/linear/" +
                          model_name};
  const std::string params{"/shixiaowei02/Paddle-InfRT/Paddle/linear/" +
                           param_name};

  paddle_infer::Config config;
  config.SwitchIrOptim(false);
  config.SetModel(model, params);
  auto predictor = paddle_infer::CreatePredictor(config);

  std::vector<int> input_shape({16, 784});
  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->mutable_data<float>(paddle_infer::PlaceType::kCPU);

  for (size_t i = 0; i < 100; ++i) {
    predictor->Run();
  }

  Timer timer;
  timer.tic();
  const size_t times = 100000;
  for (size_t i = 0; i < times; ++i) {
    predictor->Run();
  }
  double time = timer.toc() / times;
  std::cout << "time = " << time << '\n';
}

int main() {
  for (auto layers : {1, 10, 50}) {
    for (auto num : {1, 100, 1000, 10000}) {
      benchmark(layers, num);
    }
  }
  return 0;
}
