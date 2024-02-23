/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <numeric>

#include "paddle/common/flags.h"
#include "test/cpp/inference/api/trt_test_helper.h"

namespace paddle {
namespace inference {

TEST(quant_int8, resnet50) {
  std::string model_dir = FLAGS_infer_model;
  AnalysisConfig config;
  config.EnableUseGpu(1000, 0);
  config.SetModel(model_dir);
  config.EnableTensorRtEngine(
      1 << 30, 1, 1, AnalysisConfig::Precision::kInt8, false, false);
  std::map<std::string, std::vector<int>> min_input_shape = {
      {"image", {1, 1, 3, 3}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {"image", {1, 1, 10, 10}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {"image", {1, 1, 3, 3}}};

  config.SetTRTDynamicShapeInfo(
      min_input_shape, max_input_shape, opt_input_shape);
  auto predictor = CreatePaddlePredictor(config);
  auto input_names = predictor->GetInputNames();
  int channels = 1;
  int height = 3;
  int width = 3;
  int input_num = channels * height * width * 1;

  float *input = new float[input_num];
  memset(input, 0, input_num * sizeof(float));
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({1, channels, height, width});
  input_t->copy_from_cpu(input);

  ASSERT_TRUE(predictor->ZeroCopyRun());

  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());
}

}  // namespace inference
}  // namespace paddle
