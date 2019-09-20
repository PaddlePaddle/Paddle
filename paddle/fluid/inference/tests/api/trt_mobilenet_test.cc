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

#include "paddle/fluid/inference/tests/api/trt_test_helper.h"

namespace paddle {
namespace inference {

TEST(TensorRT_mobilenet, compare) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  compare(model_dir, /* use_tensorrt */ true);
  // Open it when need.
  // profile(model_dir, /* use_analysis */ true, FLAGS_use_tensorrt);
}

TEST(AnalysisPredictor, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/" + "mobilenet";
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.EnableCUDNN();
  config.SetModel(model_dir);
  config.pass_builder()->TurnOnDebug();

  std::vector<std::vector<PaddleTensor>> inputs_all;
  auto predictor = CreatePaddlePredictor(config);
  SetFakeImageInput(&inputs_all, model_dir, false, "__model__", "");

  std::vector<PaddleTensor> outputs;
  for (auto& input : inputs_all) {
    ASSERT_TRUE(predictor->Run(input, &outputs));
  }
}

}  // namespace inference
}  // namespace paddle
