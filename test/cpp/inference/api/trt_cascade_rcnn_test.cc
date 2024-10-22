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

TEST(TensorRT, cascade_rcnn) {
  std::string model_dir = FLAGS_infer_model + "/cascade_rcnn";
  AnalysisConfig config;
  int batch_size = 1;
  config.EnableNewIR(false);
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableTensorRtEngine(
      1 << 30, batch_size, 40, AnalysisConfig::Precision::kFloat32, false);

  auto predictor = CreatePaddlePredictor(config);

  int channels = 3;
  int height = 640;
  int width = 640;
  int input_num = batch_size * channels * height * width;
  float *input = new float[input_num];
  memset(input, 1.0, input_num * sizeof(float));

  float *im_shape = new float[3];
  im_shape[0] = 3.0;
  im_shape[1] = 640.0;
  im_shape[2] = 640.0;

  auto input_names = predictor->GetInputNames();

  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->copy_from_cpu(input);

  auto input_t1 = predictor->GetInputTensor(input_names[1]);
  input_t1->Reshape({batch_size, 3});
  input_t1->copy_from_cpu(im_shape);

  ASSERT_TRUE(predictor->ZeroCopyRun());
}

}  // namespace inference
}  // namespace paddle
