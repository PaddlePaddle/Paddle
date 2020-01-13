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
  std::string model_dir = FLAGS_infer_model + "/" + "mul_model";
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir);
  config.EnableLiteEngine(paddle::AnalysisConfig::Precision::kFloat32);

  std::vector<PaddleTensor> inputs;
  auto predictor = CreatePaddlePredictor(config);
  std::vector<float> input({1});

  PaddleTensor in;
  in.shape = {1, 1};
  in.data = PaddleBuf(static_cast<void*>(input.data()), 1 * sizeof(float));
  in.dtype = PaddleDType::FLOAT32;
  inputs.emplace_back(in);

  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(inputs, &outputs));

  const std::vector<float> truth_values = {
      -0.00621776, -0.00620937, 0.00990623,  -0.0039817, -0.00074315,
      0.61229795,  -0.00491806, -0.00068755, 0.18409646, 0.30090684};

  const size_t expected_size = 1;
  EXPECT_EQ(outputs.size(), expected_size);
  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); ++j) {
    EXPECT_LT(std::abs(data_o[j] - truth_values[j]), 10e-6);
  }
}

}  // namespace inference
}  // namespace paddle
