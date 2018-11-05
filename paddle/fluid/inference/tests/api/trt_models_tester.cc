// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
using paddle::contrib::AnalysisConfig;

DEFINE_string(dirname, "", "Directory of the inference model.");

NativeConfig GetConfigNative() {
  NativeConfig config;
  config.model_dir = FLAGS_dirname;
  // LOG(INFO) << "dirname  " << config.model_dir;
  config.fraction_of_gpu_memory = 0.15;
  config.use_gpu = true;
  config.device = 0;
  return config;
}

void PrepareTRTConfig(AnalysisConfig *config) {
  config->model_dir = FLAGS_dirname + "/" + "mobilenet";
  config->fraction_of_gpu_memory = 0.15;
  config->EnableTensorRtEngine(1 << 10, 5);
  config->pass_builder()->DeletePass("conv_bn_fuse_pass");
  config->pass_builder()->DeletePass("fc_fuse_pass");
  config->pass_builder()->TurnOnDebug();
}

void PrepareInputs(std::vector<PaddleTensor> *tensors, int batch_size) {
  PADDLE_ENFORCE_EQ(tensors->size(), 1UL);
  auto &tensor = tensors->front();
  int height = 224;
  int width = 224;
  float *data = new float[batch_size * 3 * height * width];
  memset(data, 0, sizeof(float) * (batch_size * 3 * height * width));
  data[0] = 1.0f;

  // Prepare inputs
  tensor.name = "input_0";
  tensor.shape = std::vector<int>({batch_size, 3, height, width});
  tensor.data = PaddleBuf(static_cast<void *>(data),
                          sizeof(float) * (batch_size * 3 * height * width));
  tensor.dtype = PaddleDType::FLOAT32;
}

void CompareTensorRTWithFluid(int batch_size, std::string model_dirname) {
  auto config0 = GetConfigNative();
  config0.model_dir = model_dirname;

  AnalysisConfig config1(true);
  PrepareTRTConfig(&config1);
  config1.model_dir = model_dirname;

  auto predictor0 = CreatePaddlePredictor<NativeConfig>(config0);
  auto predictor1 = CreatePaddlePredictor(config1);

  // Prepare inputs
  std::vector<PaddleTensor> paddle_tensor_feeds(1);
  PrepareInputs(&paddle_tensor_feeds, batch_size);

  // Prepare outputs
  std::vector<PaddleTensor> outputs0;
  std::vector<PaddleTensor> outputs1;
  CHECK(predictor0->Run(paddle_tensor_feeds, &outputs0));
  CHECK(predictor1->Run(paddle_tensor_feeds, &outputs1, batch_size));

  const size_t num_elements = outputs0.front().data.length() / sizeof(float);
  const size_t num_elements1 = outputs1.front().data.length() / sizeof(float);
  EXPECT_EQ(num_elements, num_elements1);

  auto *data0 = static_cast<float *>(outputs0.front().data.data());
  auto *data1 = static_cast<float *>(outputs1.front().data.data());

  ASSERT_GT(num_elements, 0UL);
  for (size_t i = 0; i < std::min(num_elements, num_elements1); i++) {
    EXPECT_NEAR(data0[i], data1[i], 1e-3);
  }
}

TEST(trt_models_test, mobilenet) {
  CompareTensorRTWithFluid(1, FLAGS_dirname + "/" + "mobilenet");
}
TEST(trt_models_test, resnet50) {
  CompareTensorRTWithFluid(1, FLAGS_dirname + "/" + "resnet50");
}
TEST(trt_models_test, resnext50) {
  CompareTensorRTWithFluid(1, FLAGS_dirname + "/" + "resnext50");
}

TEST(Analyzer, use_gpu) {
  AnalysisConfig config(false);
  config.model_dir = FLAGS_dirname + "/" + "mobilenet";
  config.fraction_of_gpu_memory = 0.1;
  config.device = 0;
  config.enable_ir_optim = true;
  config.pass_builder()->TurnOnDebug();
  // config.EnableTensorRtEngine();

  auto predictor = CreatePaddlePredictor<AnalysisConfig>(config);
  // auto base_predictor = CreatePaddlePredictor<NativeConfig>(config);

  std::vector<PaddleTensor> inputs(1);
  PrepareInputs(&inputs, 2);

  std::vector<PaddleTensor> outputs;
  inference::Timer timer;

  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    ASSERT_TRUE(predictor->Run(inputs, &outputs));
  }
  LOG(INFO) << "analysis latency: " << timer.toc() / 10 << " ms";

  int num_ops{0};
  auto fuse_statis = inference::GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);

  // ASSERT_EQ(fuse_statis["conv_bn_fuse"], 14);
  // ASSERT_EQ(fuse_statis["original_graph"],
  // 87);  // not eq if the model is changed.
}

}  // namespace paddle

USE_PASS(tensorrt_subgraph_pass);
