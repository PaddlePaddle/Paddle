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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstdio>

#include "gflags/gflags.h"
#include "paddle/fluid/inference/tests/api/trt_test_helper.h"

namespace paddle {
namespace inference {

static bool FileExists(const std::string &filepath) {
  std::ifstream file(filepath);
  bool exists = file.is_open();
  file.close();
  return exists;
}

TEST(AnalysisPredictor, trt_static_engine) {
  std::string model_dir = FLAGS_infer_model + "/" + "mobilenet";
  std::string static_path = FLAGS_infer_model + "/trt_engine";
  if (FileExists(static_path)) remove(static_path.c_str());

  // step 1: collect shape info
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir);
  config.CollectShapeRangeInfo("shape_range.pbtxt");

  std::vector<std::vector<PaddleTensor>> inputs_all;
  auto predictor = CreatePaddlePredictor(config);
  SetFakeImageInput(&inputs_all, model_dir, false, "__model__", "");

  std::vector<PaddleTensor> outputs;
  for (auto &input : inputs_all) {
    ASSERT_TRUE(predictor->Run(input, &outputs));
    predictor->ClearIntermediateTensor();
  }
  // delete predictor to save shape_range.pbtxt
  delete predictor;

  // step 2: build and save TensorRT engine
  AnalysisConfig config2;
  config2.EnableUseGpu(100, 0);
  config2.SetModel(model_dir);
  config2.EnableTunedTensorRtDynamicShape("shape_range.pbtxt", true);
  config2.EnableTensorRtEngine(1 << 30,
                               1,
                               3,
                               AnalysisConfig::Precision::kFloat32,
                               true,
                               false,
                               static_path);
  auto predictor2 = CreatePaddlePredictor(config2);
  for (auto &input : inputs_all) {
    ASSERT_TRUE(predictor2->Run(input, &outputs));
    predictor2->ClearIntermediateTensor();
  }

  // step 3: load TensorRT engine
  AnalysisConfig config3;
  config3.EnableUseGpu(100, 0);
  config3.SetModel(model_dir);
  config3.EnableTunedTensorRtDynamicShape("shape_range.pbtxt", true);
  config3.EnableTensorRtEngine(1 << 30,
                               1,
                               3,
                               AnalysisConfig::Precision::kFloat32,
                               true,
                               false,
                               static_path);
  auto predictor3 = CreatePaddlePredictor(config3);
  for (auto &input : inputs_all) {
    ASSERT_TRUE(predictor3->Run(input, &outputs));
    predictor3->ClearIntermediateTensor();
  }
}

}  // namespace inference
}  // namespace paddle
