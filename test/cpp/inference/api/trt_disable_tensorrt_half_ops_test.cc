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

#include "test/cpp/inference/api/trt_test_helper.h"

namespace paddle {
namespace inference {

TEST(TensorRT, disable_tensorrt_half_ops) {
  std::string model_dir = FLAGS_infer_model + "/resnet50";
  AnalysisConfig config;
  config.SetModel(model_dir);
  config.EnableUseGpu(100, 0);
  config.EnableTensorRtEngine(
      1 << 30, 1, 5, AnalysisConfig::Precision::kHalf, false, false);

  paddle_infer::experimental::InternalUtils::DisableTensorRtHalfOps(&config,
                                                                    {"conv2d"});

  std::vector<std::vector<PaddleTensor>> inputs_all;
  auto predictor = CreatePaddlePredictor(config);
  SetFakeImageInput(&inputs_all, model_dir, false, "__model__", "");

  std::vector<PaddleTensor> outputs;
  for (auto &input : inputs_all) {
    ASSERT_TRUE(predictor->Run(input, &outputs));
    predictor->ClearIntermediateTensor();
  }
}

}  // namespace inference
}  // namespace paddle
