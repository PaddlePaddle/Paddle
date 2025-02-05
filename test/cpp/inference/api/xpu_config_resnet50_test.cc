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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cmath>
#include "paddle/common/flags.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle_infer {

static const std::vector<float> TRUTH_VALUES = {
    127.779f,  738.165f,  1013.22f,  -438.17f,  366.401f,  927.659f,  736.222f,
    -633.684f, -329.927f, -430.155f, -633.062f, -146.548f, -1324.28f, -1349.36f,
    -242.675f, 117.448f,  -801.723f, -391.514f, -404.818f, 454.16f,   515.48f,
    -133.031f, 69.293f,   590.096f,  -1434.69f, -1070.89f, 307.074f,  400.525f,
    -316.12f,  -587.125f, -161.056f, 800.363f,  -96.4708f, 748.706f,  868.174f,
    -447.938f, 112.737f,  1127.2f,   47.4355f,  677.72f,   593.186f,  -336.4f,
    551.362f,  397.823f,  78.3979f,  -715.398f, 405.969f,  404.256f,  246.019f,
    -8.42969f, 131.365f,  -648.051f};

void PrepareInput(std::shared_ptr<Predictor> predictor) {
  const int batch = 1;
  const int channel = 3;
  const int height = 318;
  const int width = 318;
  const int input_num = batch * channel * height * width;
  std::vector<float> input(input_num, 1);
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape({batch, channel, height, width});
  input_t->CopyFromCpu(input.data());
}

void CompareOutput(std::shared_ptr<Predictor> predictor) {
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  size_t out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  std::vector<float> out_data;
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());

  float* data_o = out_data.data();
  for (size_t j = 0; j < out_num; j += 10) {
    EXPECT_NEAR(
        (data_o[j] - TRUTH_VALUES[j / 10]) / TRUTH_VALUES[j / 10], 0., 10e-3);
  }
}

TEST(xpu_config, inference) {
  size_t l3_size = 10 * 1024 * 1024;
  XpuConfig xpu_config;
  xpu_config.l3_size = l3_size;
  std::string model_dir = FLAGS_infer_model + "/" + "model";
  Config config;
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableXpu();
  config.SetXpuConfig(xpu_config);

  XpuConfig xpu_config_test = config.xpu_config();
  PADDLE_ENFORCE_EQ(xpu_config_test.l3_size,
                    l3_size,
                    common::errors::InvalidArgument(
                        "xpu_config_test.l3_size %d is different from our "
                        "expected value l3_size %d.",
                        xpu_config_test.l3_size,
                        l3_size));

  auto predictor = CreatePredictor(config);
  PrepareInput(predictor);
  predictor->Run();
  CompareOutput(predictor);
}

}  // namespace paddle_infer
