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
#include <cmath>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {

int test_main(const AnalysisConfig& config, Barrier* barrier = nullptr) {
  static std::mutex mutex;
  std::unique_ptr<PaddlePredictor> predictor;
  {
    std::unique_lock<std::mutex> lock(mutex);
    predictor = std::move(CreatePaddlePredictor(config));
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
  const std::vector<float> truth_values = {
      -0.00621776, -0.00620937, 0.00990623,  -0.0039817, -0.00074315,
      0.61229795,  -0.00491806, -0.00068755, 0.18409646, 0.30090684};
  const size_t expected_size = 1;
  EXPECT_EQ(outputs.size(), expected_size);
  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); ++j) {
    EXPECT_LT(std::abs(data_o[j] - truth_values[j]), 10e-6);
  }
  return 0;
}

#ifdef PADDLE_WITH_CUDA
TEST(AnalysisPredictor, thread_local_stream) {
  const size_t thread_num = 5;
  std::vector<std::thread> threads(thread_num);
  Barrier barrier(thread_num);
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&barrier, i]() {
      AnalysisConfig config;
      config.EnableUseGpu(100, 0);
      config.SetModel(FLAGS_infer_model + "/" + "mul_model");
      config.EnableGpuMultiStream();
      test_main(config, &barrier);
    });
  }
  for (auto& th : threads) {
    th.join();
  }
}

TEST(AnalysisPredictor, lite_engine) {
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel(FLAGS_infer_model + "/" + "mul_model");
  config.EnableLiteEngine(paddle::AnalysisConfig::Precision::kFloat32);
  test_main(config);
}
#endif

}  // namespace inference
}  // namespace paddle
