/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>

#include "paddle/common/flags.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {

static std::vector<float> truth_values = {
    127.779f,  738.165f,  1013.22f,  -438.17f,  366.401f,  927.659f,  736.222f,
    -633.684f, -329.927f, -430.155f, -633.062f, -146.548f, -1324.28f, -1349.36f,
    -242.675f, 117.448f,  -801.723f, -391.514f, -404.818f, 454.16f,   515.48f,
    -133.031f, 69.293f,   590.096f,  -1434.69f, -1070.89f, 307.074f,  400.525f,
    -316.12f,  -587.125f, -161.056f, 800.363f,  -96.4708f, 748.706f,  868.174f,
    -447.938f, 112.737f,  1127.2f,   47.4355f,  677.72f,   593.186f,  -336.4f,
    551.362f,  397.823f,  78.3979f,  -715.398f, 405.969f,  404.256f,  246.019f,
    -8.42969f, 131.365f,  -648.051f};

// Compare results with 1 batch
TEST(Analyzer_Resnet50_ipu, compare_results_1_batch) {
  std::string model_dir = FLAGS_infer_model + "/" + "model";
  AnalysisConfig config;
  // ipu_device_num, ipu_micro_batch_size, ipu_enable_pipelining
  config.EnableIpu(1, 1, false);
  config.SetModel(model_dir + "/model", model_dir + "/params");

  std::vector<PaddleTensor> inputs;
  auto predictor = CreatePaddlePredictor(config);
  const int batch = 1;
  const int channel = 3;
  const int height = 318;
  const int width = 318;
  const int input_num = batch * channel * height * width;
  std::vector<float> input(input_num, 1);

  PaddleTensor in;
  in.shape = {batch, channel, height, width};
  in.data =
      PaddleBuf(static_cast<void*>(input.data()), input_num * sizeof(float));
  in.dtype = PaddleDType::FLOAT32;
  inputs.emplace_back(in);

  std::vector<PaddleTensor> outputs;

  ASSERT_TRUE(predictor->Run(inputs, &outputs));

  const size_t expected_size = 1;
  EXPECT_EQ(outputs.size(), expected_size);
  float* data_o = static_cast<float*>(outputs[0].data.data());

  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); j += 10) {
    EXPECT_NEAR(
        (data_o[j] - truth_values[j / 10]) / truth_values[j / 10], 0., 12e-5);
  }
}

// Compare results with 2 batch
TEST(Analyzer_Resnet50_ipu, compare_results_2_batch) {
  std::string model_dir = FLAGS_infer_model + "/" + "model";
  AnalysisConfig config;
  // ipu_device_num, ipu_micro_batch_size, ipu_enable_pipelining
  config.EnableIpu(1, 2, false);
  config.SetModel(model_dir + "/model", model_dir + "/params");

  std::vector<PaddleTensor> inputs;
  auto predictor = CreatePaddlePredictor(config);
  const int batch = 2;
  const int channel = 3;
  const int height = 318;
  const int width = 318;
  const int input_num = batch * channel * height * width;
  std::vector<float> input(input_num, 1);

  PaddleTensor in;
  in.shape = {batch, channel, height, width};
  in.data =
      PaddleBuf(static_cast<void*>(input.data()), input_num * sizeof(float));
  in.dtype = PaddleDType::FLOAT32;
  inputs.emplace_back(in);

  std::vector<PaddleTensor> outputs;

  ASSERT_TRUE(predictor->Run(inputs, &outputs));

  const size_t expected_size = 1;
  EXPECT_EQ(outputs.size(), expected_size);
  float* data_o = static_cast<float*>(outputs[0].data.data());

  auto num_output_per_batch = outputs[0].data.length() / sizeof(float) / 2;
  for (size_t j = 0; j < num_output_per_batch; j += 10) {
    EXPECT_NEAR(
        (data_o[j] - truth_values[j / 10]) / truth_values[j / 10], 0., 12e-5);
    EXPECT_NEAR((data_o[j + num_output_per_batch] - truth_values[j / 10]) /
                    truth_values[j / 10],
                0.,
                12e-5);
  }
}

// multi threading
TEST(Analyzer_Resnet50_ipu, model_runtime_multi_thread) {
  std::string model_dir = FLAGS_infer_model + "/" + "model";
  AnalysisConfig config;
  const int thread_num = 10;
  // ipu_device_num, ipu_micro_batch_size, ipu_enable_pipelining
  config.EnableIpu(1, 1, false);
  config.SetIpuConfig(false, 1, 1.0, false, true);
  config.SetModel(model_dir + "/model", model_dir + "/params");

  auto main_predictor = CreatePaddlePredictor(config);
  std::vector<std::vector<PaddleTensor>> inputs;
  std::vector<std::vector<PaddleTensor>> outputs;
  std::vector<decltype(main_predictor)> predictors;
  std::vector<std::thread> threads;
  outputs.resize(thread_num);
  inputs.resize(thread_num);

  const int batch = 1;
  const int channel = 3;
  const int height = 318;
  const int width = 318;
  const int input_num = batch * channel * height * width;
  std::vector<float> input(input_num, 1);

  PaddleTensor in;
  in.shape = {batch, channel, height, width};
  in.data =
      PaddleBuf(static_cast<void*>(input.data()), input_num * sizeof(float));
  in.dtype = PaddleDType::FLOAT32;

  for (int i = 0; i < thread_num; ++i) {
    inputs[i].emplace_back(in);
    predictors.emplace_back(std::move(main_predictor->Clone()));
  }

  auto run = [](PaddlePredictor* predictor,
                std::vector<PaddleTensor>& input,
                std::vector<PaddleTensor>& output) {
    ASSERT_TRUE(predictor->Run(input, &output));
  };

  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(
        run, predictors[i].get(), std::ref(inputs[i]), std::ref(outputs[i]));
  }

  for (int i = 0; i < thread_num; ++i) {
    threads[i].join();
  }

  const size_t expected_size = 1;
  for (int i = 0; i < thread_num; ++i) {
    EXPECT_EQ(outputs[i].size(), expected_size);
    float* data_o = static_cast<float*>(outputs[i][0].data.data());

    for (size_t j = 0; j < outputs[i][0].data.length() / sizeof(float);
         j += 10) {
      EXPECT_NEAR(
          (data_o[j] - truth_values[j / 10]) / truth_values[j / 10], 0., 12e-5);
    }
  }
}
}  // namespace inference
}  // namespace paddle
