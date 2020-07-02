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
  config.EnableLiteEngine(paddle::AnalysisConfig::Precision::kFloat32);

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
      127.780396f, 738.16656f,  1013.2264f,  -438.17206f, 366.4022f,
      927.66187f,  736.2241f,   -633.68567f, -329.92737f, -430.15637f,
      -633.0639f,  -146.54858f, -1324.2804f, -1349.3661f, -242.67671f,
      117.44864f,  -801.7251f,  -391.51495f, -404.8202f,  454.16132f,
      515.48206f,  -133.03114f, 69.293076f,  590.09753f,  -1434.6917f,
      -1070.8903f, 307.0744f,   400.52573f,  -316.12177f, -587.1265f,
      -161.05742f, 800.3663f,   -96.47157f,  748.708f,    868.17645f,
      -447.9403f,  112.73656f,  1127.1992f,  47.43518f,   677.7219f,
      593.1881f,   -336.4011f,  551.3634f,   397.82474f,  78.39835f,
      -715.4006f,  405.96988f,  404.25684f,  246.01978f,  -8.430191f,
      131.36617f,  -648.0528f};

  const size_t expected_size = 1;
  EXPECT_EQ(outputs.size(), expected_size);
  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); j += 10) {
    EXPECT_NEAR((data_o[j] - truth_values[j / 10]) / truth_values[j / 10], 0.,
                10e-5);
  }
}

}  // namespace inference
}  // namespace paddle
