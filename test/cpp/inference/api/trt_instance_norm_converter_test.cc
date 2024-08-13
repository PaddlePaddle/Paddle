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

#include "paddle/common/flags.h"
#include "test/cpp/inference/api/trt_test_helper.h"

namespace paddle {
namespace inference {

TEST(TensorRT, instance_norm) {
  std::string model_dir = FLAGS_infer_model + "/instance_norm";
  AnalysisConfig config;
  int batch_size = 4;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir);
  config.EnableTensorRtEngine(
      1 << 20, batch_size, 0, AnalysisConfig::Precision::kFloat32, false);

  auto predictor = CreatePaddlePredictor(config);

  int length = 4;
  int input_num = batch_size * length;
  float *input = new float[input_num];
  memset(input, 1.0, input_num * sizeof(float));

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, length});
  input_t->copy_from_cpu(input);

  ASSERT_TRUE(predictor->ZeroCopyRun());
}

}  // namespace inference
}  // namespace paddle
