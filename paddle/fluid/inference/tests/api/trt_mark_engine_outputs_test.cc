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

#include "gflags/gflags.h"
#include "paddle/fluid/inference/tests/api/trt_test_helper.h"

namespace paddle {
namespace inference {
TEST(TensorRT, mark_engine_output) {
  std::string model_dir = FLAGS_infer_model;
  // The name of the tensor that needs to be marked, the default is empty (all
  // marks)
  std::vector<std::string> markOutput = {};
  AnalysisConfig config;
  config.SetModel(model_dir);
  config.EnableUseGpu(500, 0);
  config.EnableTensorRtEngine(
      1 << 30, 1, 5, AnalysisConfig::Precision::kFloat32, false, false);
  config.MarkEngineOutputs(true, markOutput);
  auto predictor = CreatePaddlePredictor(config);

  int batch_size = 1;
  int channels = 3;
  int height = 324;
  int width = 324;
  int input_num = batch_size * channels * height * width;
  float *input = new float[input_num];
  memset(input, 1.0, input_num * sizeof(float));

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->copy_from_cpu(input);

  ASSERT_TRUE(predictor->ZeroCopyRun());
}
}  // namespace inference
}  // namespace paddle
