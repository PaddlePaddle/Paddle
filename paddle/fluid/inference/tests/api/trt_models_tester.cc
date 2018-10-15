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
using paddle::contrib::MixedRTConfig;
using paddle::contrib::AnalysisConfig;

DEFINE_string(dirname, "", "Directory of the inference model.");

NativeConfig GetConfigNative() {
  NativeConfig config;
  config.model_dir = FLAGS_dirname;
  // LOG(INFO) << "dirname  " << config.model_dir;
  config.fraction_of_gpu_memory = 0.45;
  config.use_gpu = true;
  config.device = 0;
  return config;
}

MixedRTConfig GetConfigTRT() {
  MixedRTConfig config;
  config.model_dir = FLAGS_dirname;
  config.use_gpu = true;
  config.fraction_of_gpu_memory = 0.2;
  config.device = 0;
  config.max_batch_size = 3;
  return config;
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
  NativeConfig config0 = GetConfigNative();
  config0.model_dir = model_dirname;

  MixedRTConfig config1 = GetConfigTRT();
  config1.model_dir = model_dirname;
  config1.max_batch_size = batch_size;

  auto predictor0 = CreatePaddlePredictor<NativeConfig>(config0);
  auto predictor1 = CreatePaddlePredictor<MixedRTConfig>(config1);

  std::vector<PaddleTensor> paddle_tensor_feeds(1);
  PrepareInputs(&paddle_tensor_feeds, batch_size);

  // Prepare outputs
  std::vector<PaddleTensor> outputs0;
  std::vector<PaddleTensor> outputs1;
  CHECK(predictor0->Run(paddle_tensor_feeds, &outputs0));

  CHECK(predictor1->Run(paddle_tensor_feeds, &outputs1, batch_size));

  // Get output.
  ASSERT_EQ(outputs0.size(), 1UL);
  ASSERT_EQ(outputs1.size(), 1UL);

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

TEST(trt_models_test, main) {
  std::vector<std::string> infer_models = {"mobilenet", "resnet50",
                                           "resnext50"};
  for (auto &model_dir : infer_models) {
    CompareTensorRTWithFluid(1, FLAGS_dirname + "/" + model_dir);
  }
}

TEST(Analyzer, use_gpu) {
  AnalysisConfig config(true);
  config.model_dir = FLAGS_dirname + "/" + "mobilenet";
  config.fraction_of_gpu_memory = 0.1;
  config.device = 0;
  config.enable_ir_optim = true;

  auto predictor = CreatePaddlePredictor<AnalysisConfig>(config);
  auto base_predictor = CreatePaddlePredictor<NativeConfig>(config);

  std::vector<PaddleTensor> inputs(1);
  PrepareInputs(&inputs, 2 /*batch size*/);

  std::vector<PaddleTensor> outputs;
  inference::Timer timer;
  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    ASSERT_TRUE(predictor->Run(inputs, &outputs));
  }
  LOG(INFO) << "analysis latency: " << timer.toc() / 10 << " ms";

  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    ASSERT_TRUE(base_predictor->Run(inputs, &outputs));
  }
  LOG(INFO) << "native latency: " << timer.toc() / 10 << " ms";

  int num_ops{0};
  auto fuse_statis = inference::GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);

  ASSERT_EQ(fuse_statis["conv_bn_fuse"], 14);
  ASSERT_EQ(fuse_statis["original_graph"],
            87);  // not eq if the model is changed.
}

}  // namespace paddle
