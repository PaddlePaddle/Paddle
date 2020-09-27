/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cmath>

#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {

TEST(AnalysisPredictor, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/" + "model";
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableLiteEngine(paddle::AnalysisConfig::Precision::kFloat32, true);

  std::vector<PaddleTensor> inputs;
  auto predictor = CreatePaddlePredictor(config);
  const int batch = 1;
  const int channel = 3;
  const int height = 318;
  const int width = 318;
  const int input_num = batch * channel * height * width;
  std::vector<float> input(input_num, 1);

  PaddleTensor in;
  in.shape = {1, 3, 318, 318};
  in.data =
      PaddleBuf(static_cast<void*>(input.data()), input_num * sizeof(float));
  in.dtype = PaddleDType::FLOAT32;
  inputs.emplace_back(in);

  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(inputs, &outputs));

  const std::vector<float> truth_values = {
      127.779,  738.165,  1013.22,  -438.17,  366.401,  927.659,  736.222,
      -633.684, -329.927, -430.155, -633.062, -146.548, -1324.28, -1349.36,
      -242.675, 117.448,  -801.723, -391.514, -404.818, 454.16,   515.48,
      -133.031, 69.293,   590.096,  -1434.69, -1070.89, 307.074,  400.525,
      -316.12,  -587.125, -161.056, 800.363,  -96.4708, 748.706,  868.174,
      -447.938, 112.737,  1127.2,   47.4355,  677.72,   593.186,  -336.4,
      551.362,  397.823,  78.3979,  -715.398, 405.969,  404.256,  246.019,
      -8.42969, 131.365,  -648.051};

  const size_t expected_size = 1;
  EXPECT_EQ(outputs.size(), expected_size);
  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); j += 10) {
    EXPECT_NEAR((data_o[j] - truth_values[j / 10]) / truth_values[j / 10], 0.,
                12e-5);
  }
}

}  // namespace inference
}  // namespace paddle

namespace paddle_infer {

TEST(Predictor, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/" + "model";
  Config config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableLiteEngine(PrecisionType::kFloat32);

  auto predictor = CreatePredictor(config);
  const int batch = 1;
  const int channel = 3;
  const int height = 318;
  const int width = 318;
  const int input_num = batch * channel * height * width;
  std::vector<float> input(input_num, 1);

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);

  input_t->Reshape({1, 3, 318, 318});
  input_t->CopyFromCpu(input.data());
  predictor->Run();

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  size_t out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                   std::multiplies<int>());

  std::vector<float> out_data;
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());

  const std::vector<float> truth_values = {
      127.779,  738.165,  1013.22,  -438.17,  366.401,  927.659,  736.222,
      -633.684, -329.927, -430.155, -633.062, -146.548, -1324.28, -1349.36,
      -242.675, 117.448,  -801.723, -391.514, -404.818, 454.16,   515.48,
      -133.031, 69.293,   590.096,  -1434.69, -1070.89, 307.074,  400.525,
      -316.12,  -587.125, -161.056, 800.363,  -96.4708, 748.706,  868.174,
      -447.938, 112.737,  1127.2,   47.4355,  677.72,   593.186,  -336.4,
      551.362,  397.823,  78.3979,  -715.398, 405.969,  404.256,  246.019,
      -8.42969, 131.365,  -648.051};

  float* data_o = out_data.data();
  for (size_t j = 0; j < out_num; j += 10) {
    EXPECT_NEAR((data_o[j] - truth_values[j / 10]) / truth_values[j / 10], 0.,
                10e-5);
  }
}

}  // namespace paddle_infer
