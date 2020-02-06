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
      127.780396, 738.16656,  1013.2264,  -438.17206, 366.4022,   927.66187,
      736.2241,   -633.68567, -329.92737, -430.15637, -633.0639,  -146.54858,
      -1324.2804, -1349.3661, -242.67671, 117.44864,  -801.7251,  -391.51495,
      -404.8202,  454.16132,  515.48206,  -133.03114, 69.293076,  590.09753,
      -1434.6917, -1070.8903, 307.0744,   400.52573,  -316.12177, -587.1265,
      -161.05742, 800.3663,   -96.47157,  748.708,    868.17645,  -447.9403,
      112.73656,  1127.1992,  47.43518,   677.7219,   593.1881,   -336.4011,
      551.3634,   397.82474,  78.39835,   -715.4006,  405.96988,  404.25684,
      246.01978,  -8.430191,  131.36617,  -648.0528};

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
