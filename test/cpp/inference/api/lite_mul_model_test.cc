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

#include <cmath>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "paddle/common/flags.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {

int test_predictor(const AnalysisConfig& config_in,
                   Barrier* barrier = nullptr) {
  static std::mutex mutex;
  AnalysisConfig config{config_in};
  std::unique_ptr<PaddlePredictor> predictor;
  {
    std::unique_lock<std::mutex> lock(mutex);
    predictor = CreatePaddlePredictor(config);
  }
  if (barrier) {
    barrier->Wait();
  }

  std::vector<PaddleTensor> inputs;
  std::vector<float> input({1});

  PaddleTensor in;
  in.shape = {1, 1};
  in.data = PaddleBuf(static_cast<void*>(input.data()), 1 * sizeof(float));
  in.dtype = PaddleDType::FLOAT32;
  inputs.emplace_back(in);

  std::vector<PaddleTensor> outputs;
  predictor->Run(inputs, &outputs);
  const std::vector<float> truth_values = {-0.00621776f,
                                           -0.00620937f,
                                           0.00990623f,
                                           -0.0039817f,
                                           -0.00074315f,
                                           0.61229795f,
                                           -0.00491806f,
                                           -0.00068755f,
                                           0.18409646f,
                                           0.30090684f};
  const size_t expected_size = 1;
  EXPECT_EQ(outputs.size(), expected_size);
  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); ++j) {
    EXPECT_LT(std::abs(data_o[j] - truth_values[j]), 10e-6);
  }
  return 0;
}

int test_predictor_zero_copy(const AnalysisConfig& config_in,
                             Barrier* barrier = nullptr) {
  static std::mutex mutex;
  AnalysisConfig config{config_in};
  std::unique_ptr<PaddlePredictor> predictor;
  {
    std::unique_lock<std::mutex> lock(mutex);
    predictor = CreatePaddlePredictor(config);
  }
  if (barrier) {
    barrier->Wait();
  }

  std::vector<float> input({1});
  auto in_tensor =
      predictor->GetInputTensor(predictor->GetInputNames().front());
  in_tensor->Reshape({1, 1});
  in_tensor->copy_from_cpu(input.data());

  predictor->ZeroCopyRun();

  auto out_tensor =
      predictor->GetOutputTensor(predictor->GetOutputNames().front());
  std::vector<float> data_o(10);
  out_tensor->copy_to_cpu(data_o.data());

  const std::vector<float> truth_values = {-0.00621776f,
                                           -0.00620937f,
                                           0.00990623f,
                                           -0.0039817f,
                                           -0.00074315f,
                                           0.61229795f,
                                           -0.00491806f,
                                           -0.00068755f,
                                           0.18409646f,
                                           0.30090684f};
  const size_t expected_size = 1;
  EXPECT_EQ(predictor->GetOutputNames().size(), expected_size);
  for (size_t j = 0; j < truth_values.size(); ++j) {
    EXPECT_LT(std::abs(data_o[j] - truth_values[j]), 10e-6);
  }
  return 0;
}

#ifdef PADDLE_WITH_XPU
TEST(AnalysisPredictor, native_xpu) {
  AnalysisConfig config;
  config.EnableXpu();
  config.SetModel(FLAGS_infer_model + "/" + "mul_model");
  test_predictor(config);
  test_predictor_zero_copy(config);
}
#endif

}  // namespace inference
}  // namespace paddle
