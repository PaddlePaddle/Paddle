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
#include <thread>

#include "paddle/common/flags.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {

// TODO(inference): This case failed in windows with a SEH error, we need to fix
// it.
TEST(ReBindStream_single, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir);
  config.EnableTensorRtEngine();

  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  config.SetExecStream(stream1);
  auto predictor = paddle_infer::CreatePredictor(config);
  auto x_t = predictor->GetInputHandle("x");
  x_t->Reshape({1, 3, 224, 224});
  std::array<float, 3 * 224 * 224> x_data = {0};
  x_t->CopyFromCpu(x_data.data());
  ASSERT_TRUE(predictor->Run());
  cudaDeviceSynchronize();
  ASSERT_TRUE(paddle_infer::experimental::InternalUtils::RunWithExternalStream(
      predictor.get(), stream2));
  cudaDeviceSynchronize();
  ASSERT_TRUE(paddle_infer::experimental::InternalUtils::RunWithExternalStream(
      predictor.get(), stream3));
  cudaDeviceSynchronize();
}

TEST(ReBindStream_multi, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  AnalysisConfig config1;
  config1.EnableUseGpu(100, 0);
  config1.SetModel(model_dir);
  config1.EnableTensorRtEngine();
  AnalysisConfig config2;
  config2.EnableUseGpu(100, 0);
  config2.EnableTensorRtEngine();
  config2.SetModel(model_dir);

  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  config1.SetExecStream(stream1);
  config2.SetExecStream(stream1);
  auto predictor1 = paddle_infer::CreatePredictor(config1);
  auto predictor2 = paddle_infer::CreatePredictor(config2);

  std::vector<float> x1(3 * 224 * 224, 1.0);
  auto x_t1 = predictor1->GetInputHandle("x");
  x_t1->Reshape({1, 3, 224, 224});
  x_t1->CopyFromCpu(x1.data());
  std::vector<float> x2(3 * 224 * 224, 2.0);
  auto x_t2 = predictor2->GetInputHandle("x");
  x_t2->Reshape({1, 3, 224, 224});
  x_t2->CopyFromCpu(x2.data());

  ASSERT_TRUE(predictor1->Run());
  cudaStreamSynchronize(stream1);
  ASSERT_TRUE(predictor2->Run());
  cudaStreamSynchronize(stream1);

  ASSERT_TRUE(paddle_infer::experimental::InternalUtils::RunWithExternalStream(
      predictor1.get(), stream2));
  cudaDeviceSynchronize();
  ASSERT_TRUE(paddle_infer::experimental::InternalUtils::RunWithExternalStream(
      predictor2.get(), stream2));
  cudaDeviceSynchronize();

  ASSERT_TRUE(paddle_infer::experimental::InternalUtils::RunWithExternalStream(
      predictor1.get(), stream3));
  cudaStreamSynchronize(stream3);
  ASSERT_TRUE(paddle_infer::experimental::InternalUtils::RunWithExternalStream(
      predictor2.get(), stream3));
  cudaStreamSynchronize(stream3);
}

TEST(SwitchStream_multi, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  AnalysisConfig config1;
  config1.EnableUseGpu(100, 0);
  config1.SetModel(model_dir);
  AnalysisConfig config2;
  config2.EnableUseGpu(100, 0);
  config2.SetModel(model_dir);
  AnalysisConfig config3;
  config3.EnableUseGpu(100, 0);
  config3.SetModel(model_dir);

  // config1.EnableTensorRtEngine();
  // config2.EnableTensorRtEngine();
  // config3.EnableTensorRtEngine();

  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  config1.SetExecStream(stream1);
  config2.SetExecStream(stream1);
  config3.SetExecStream(stream1);
  auto predictor1 = paddle_infer::CreatePredictor(config1);
  auto predictor2 = paddle_infer::CreatePredictor(config2);
  auto predictor3 = paddle_infer::CreatePredictor(config3);

  std::vector<float> x1(3 * 224 * 224, 1.0);
  auto x_t1 = predictor1->GetInputHandle("x");
  x_t1->Reshape({1, 3, 224, 224});
  x_t1->CopyFromCpu(x1.data());
  std::vector<float> x2(3 * 224 * 224, 2.0);
  auto x_t2 = predictor2->GetInputHandle("x");
  x_t2->Reshape({1, 3, 224, 224});
  x_t2->CopyFromCpu(x2.data());
  std::vector<float> x3(3 * 224 * 224, 2.5);
  auto x_t3 = predictor3->GetInputHandle("x");
  x_t3->Reshape({1, 3, 224, 224});
  x_t3->CopyFromCpu(x3.data());

  // TODO(wilber): fix.
  // NOTE: Must run once on master thread, but why?
  // if remove the code, the unit test fail.
  ASSERT_TRUE(predictor1->Run());
  cudaStreamSynchronize(stream1);
  ASSERT_TRUE(predictor2->Run());
  cudaStreamSynchronize(stream1);
  ASSERT_TRUE(predictor3->Run());
  cudaStreamSynchronize(stream1);

  auto Run = [&](paddle_infer::Predictor* p,
                 std::vector<cudaStream_t> streams) {
    for (auto s : streams) {
      paddle_infer::experimental::InternalUtils::RunWithExternalStream(p, s);
    }
  };

  std::thread p1(Run,
                 predictor1.get(),
                 std::vector<cudaStream_t>{
                     stream1, stream2, stream3, stream3, stream2, stream2});
  std::thread p2(Run,
                 predictor2.get(),
                 std::vector<cudaStream_t>{
                     stream1, stream3, stream1, stream2, stream1, stream3});
  std::thread p3(Run,
                 predictor3.get(),
                 std::vector<cudaStream_t>{
                     stream1, stream1, stream2, stream3, stream3, stream2});
  p1.join();
  p2.join();
  p3.join();
  cudaDeviceSynchronize();
}

}  // namespace inference
}  // namespace paddle
