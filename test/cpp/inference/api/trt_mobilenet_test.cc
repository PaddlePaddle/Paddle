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

#include "paddle/common/flags.h"
#include "test/cpp/inference/api/trt_test_helper.h"

namespace paddle_infer {
TEST(PredictorPool, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/" + "mobilenet";
  Config config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir);
  config.EnableTensorRtEngine();
  config.Exp_DisableTensorRtOPs({"fc"});
  config.EnableTensorRtDLA(0);
  services::PredictorPool pred_pool(config, 1);

  auto predictor = pred_pool.Retrieve(0);
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  std::vector<int> in_shape = {1, 3, 224, 224};
  int in_num =
      std::accumulate(in_shape.begin(), in_shape.end(), 1, [](int &a, int &b) {
        return a * b;
      });

  std::vector<float> input(in_num, 0);
  input_t->Reshape(in_shape);
  input_t->CopyFromCpu(input.data());
  predictor->Run();
}

TEST(PredictorPool, use_trt_cuda_graph) {
  std::string model_dir = FLAGS_infer_model + "/" + "mobilenet";
  Config config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir);
  config.EnableTensorRtEngine(
      1 << 20, 1, 3, PrecisionType::kFloat32, false, false, true);
  config.Exp_DisableTensorRtOPs({"fc"});
  config.EnableTensorRtDLA(0);
  services::PredictorPool pred_pool(config, 1);

  auto predictor = pred_pool.Retrieve(0);
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  std::vector<int> in_shape = {1, 3, 224, 224};
  int in_num =
      std::accumulate(in_shape.begin(), in_shape.end(), 1, [](int &a, int &b) {
        return a * b;
      });

  std::vector<float> input(in_num, 0);
  input_t->Reshape(in_shape);
  input_t->CopyFromCpu(input.data());
  predictor->Run();
}

}  // namespace paddle_infer
