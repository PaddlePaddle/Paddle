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

#include <gtest/gtest.h>
#include <iostream>
#include "gflags/gflags.h"

#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/infrt/tests/timer.h"
//#include "paddle/fluid/inference/api/paddle_api.h"  // NOLINT

DEFINE_int32(layers, 0, "");
DEFINE_int32(num, 0, "");

void benchmark(size_t layers, size_t num) {
  const std::string tag =
      "l" + std::to_string(layers) + "n" + std::to_string(num);
  const std::string model_name = tag + ".pdmodel";
  const std::string param_name = tag + ".pdiparams";
  const std::string prefix =
      "/shixiaowei02/Paddle-InfRT/Paddle/tools/infrt/fake_models/elt_add/";

  const std::string model{prefix + model_name};
  const std::string params{prefix + param_name};

  paddle_infer::Config config;
  config.SwitchIrOptim(false);
  config.SetModel(model, params);
  auto predictor = paddle_infer::CreatePredictor(config);

  std::vector<int> input_shape({static_cast<int>(num), static_cast<int>(num)});
  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->mutable_data<float>(paddle_infer::PlaceType::kCPU);
  float* in_p = input_t->mutable_data<float>(paddle_infer::PlaceType::kCPU);
  for (size_t i = 0; i < num * num; i++) in_p[i] = 1.0;

  predictor->Run();

  auto output_t = predictor->GetInputHandle(output_names[0]);
  int32_t numel{1};
  for (auto n : output_t->shape()) {
    numel = numel * n;
  }
  std::vector<float> output(numel);

  float* out_p = output.data();
  output_t->CopyToCpu(out_p);

  float sum{0};

  for (auto n : output) {
    sum = sum + n;
  }

  std::cout << "sum = " << sum << '\n';

  infrt::tests::BenchmarkStats timer;

  for (size_t i = 0; i < 10; ++i) {
    predictor->Run();
  }

  for (size_t j = 0; j < 100; ++j) {
    timer.Start();
    predictor->Run();
    timer.Stop();
  }
  std::cout << "\nlayers " << layers << ", num " << num << '\n';
  std::cout << "framework " << timer.Summerize({0.5});
}

TEST(InfRtPredictor, predictor) { benchmark(FLAGS_layers, FLAGS_num); }
