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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/inference/tests/api/trt_test_helper.h"

namespace paddle {
namespace inference {

TEST(TensorRT_fc, compare) {
  std::string model_dir = FLAGS_infer_model + "/fc_uint8";
  compare(model_dir, /* use_tensorrt */ true);
  // Open it when need.
  // profile(model_dir, /* use_analysis */ true, FLAGS_use_tensorrt);
}

TEST(ZeroCopyTensor, uint8) {
  std::string model_dir = FLAGS_infer_model + "/" + "fc_uint8";
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir);
  config.SwitchUseFeedFetchOps(false);
  config.EnableProfile();
  config.DisableGlogInfo();

  std::vector<std::vector<PaddleTensor>> inputs_all;
  auto predictor = CreatePaddlePredictor(config);
  auto input_names = predictor->GetInputNames();
  auto name2shape = predictor->GetInputTensorShape();

  int batch_size = 1;
  int length = 4;
  int input_num = batch_size * length;
  uint8_t *input = new uint8_t[input_num];
  memset(input, 1, input_num * sizeof(uint8_t));
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, length});
  input_t->copy_from_cpu(input);
  input_t->type();
  input_t->mutable_data<uint8_t>(PaddlePlace::kGPU);

  ASSERT_TRUE(predictor->ZeroCopyRun());
}

}  // namespace inference
}  // namespace paddle
