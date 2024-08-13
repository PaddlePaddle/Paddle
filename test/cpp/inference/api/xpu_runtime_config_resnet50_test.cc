/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "test/cpp/inference/api/tester_helper.h"
#include "xpu/runtime.h"
#include "xpu/xdnn.h"

namespace paddle_infer {

static const std::vector<float> TRUTH_VALUES = {
    127.779f,  738.165f,  1013.22f,  -438.17f,  366.401f,  927.659f,  736.222f,
    -633.684f, -329.927f, -430.155f, -633.062f, -146.548f, -1324.28f, -1349.36f,
    -242.675f, 117.448f,  -801.723f, -391.514f, -404.818f, 454.16f,   515.48f,
    -133.031f, 69.293f,   590.096f,  -1434.69f, -1070.89f, 307.074f,  400.525f,
    -316.12f,  -587.125f, -161.056f, 800.363f,  -96.4708f, 748.706f,  868.174f,
    -447.938f, 112.737f,  1127.2f,   47.4355f,  677.72f,   593.186f,  -336.4f,
    551.362f,  397.823f,  78.3979f,  -715.398f, 405.969f,  404.256f,  246.019f,
    -8.42969f, 131.365f,  -648.051f};

void PrepareInput(std::shared_ptr<Predictor> predictor) {
  const int batch = 1;
  const int channel = 3;
  const int height = 318;
  const int width = 318;
  const int input_num = batch * channel * height * width;
  std::vector<float> input(input_num, 1);
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape({batch, channel, height, width});
  input_t->CopyFromCpu(input.data());
}

void CompareOutput(std::shared_ptr<Predictor> predictor) {
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  size_t out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  std::vector<float> out_data;
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());

  float* data_o = out_data.data();
  for (size_t j = 0; j < out_num; j += 10) {
    EXPECT_NEAR(
        (data_o[j] - TRUTH_VALUES[j / 10]) / TRUTH_VALUES[j / 10], 0., 10e-3);
  }
}

Config InferXpuConfig() {
  std::string model_dir = FLAGS_infer_model + "/" + "model";
  Config config;
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableXpu();
  return config;
}

TEST(resnet50_xpu, basic) {
  Config config = InferXpuConfig();
  auto predictor = CreatePredictor(config);
  PrepareInput(predictor);
  predictor->Run();
  CompareOutput(predictor);
}

#define RUN_WITH_RUNTIME_CONFIG(idx_, config_)                             \
  Config config##idx_ = InferXpuConfig();                                  \
  auto predictor##idx_ = CreatePredictor(config##idx_);                    \
  PrepareInput(predictor##idx_);                                           \
  experimental::InternalUtils::RunWithRuntimeConfig(predictor##idx_.get(), \
                                                    &config_);             \
  CompareOutput(predictor##idx_);                                          \
  PADDLE_ENFORCE_EQ(                                                       \
      predictor##idx_->GetExecStream(),                                    \
      config_.stream,                                                      \
      common::errors::InvalidArgument(                                     \
          "predictor##idx_->GetExecStream() is not equal with"             \
          "config_.stream while predictor##idx_->GetExecStream()"          \
          "is %d and config_.stream is %d",                                \
          predictor##idx_->GetExecStream(),                                \
          config_.stream));

TEST(runtime_stream, null_stream) {
  experimental::XpuRuntimeConfig xpu_runtime_config;
  xpu_runtime_config.context = nullptr;
  xpu_runtime_config.stream = nullptr;
  xpu_runtime_config.l3_size = 0;
  xpu_runtime_config.l3_ptr = nullptr;
  xpu_runtime_config.l3_autotune_size = 0;
  RUN_WITH_RUNTIME_CONFIG(0, xpu_runtime_config);
}

TEST(runtime_stream, new_stream) {
  void* stream = nullptr;
  xpu_stream_create(&stream);
  CHECK_NOTNULL(stream);
  {
    experimental::XpuRuntimeConfig xpu_runtime_config;
    xpu_runtime_config.context = nullptr;
    xpu_runtime_config.stream = stream;
    xpu_runtime_config.l3_size = 0;
    xpu_runtime_config.l3_ptr = nullptr;
    xpu_runtime_config.l3_autotune_size = 0;
    RUN_WITH_RUNTIME_CONFIG(0, xpu_runtime_config);
  }
  xpu_stream_destroy(stream);
}

TEST(runtime_stream, 2_null_stream) {
  experimental::XpuRuntimeConfig xpu_runtime_config;
  xpu_runtime_config.context = nullptr;
  xpu_runtime_config.stream = nullptr;
  xpu_runtime_config.l3_size = 0;
  xpu_runtime_config.l3_ptr = nullptr;
  xpu_runtime_config.l3_autotune_size = 0;
  RUN_WITH_RUNTIME_CONFIG(0, xpu_runtime_config);
  RUN_WITH_RUNTIME_CONFIG(1, xpu_runtime_config);
}

TEST(runtime_stream, null_and_new_stream) {
  experimental::XpuRuntimeConfig xpu_runtime_config0;
  xpu_runtime_config0.context = nullptr;
  xpu_runtime_config0.stream = nullptr;
  xpu_runtime_config0.l3_size = 0;
  xpu_runtime_config0.l3_ptr = nullptr;
  xpu_runtime_config0.l3_autotune_size = 0;
  void* stream = nullptr;
  xpu_stream_create(&stream);
  CHECK_NOTNULL(stream);
  {
    experimental::XpuRuntimeConfig xpu_runtime_config1;
    xpu_runtime_config1.context = nullptr;
    xpu_runtime_config1.stream = stream;
    xpu_runtime_config1.l3_size = 0;
    xpu_runtime_config1.l3_ptr = nullptr;
    xpu_runtime_config1.l3_autotune_size = 0;
    RUN_WITH_RUNTIME_CONFIG(0, xpu_runtime_config0);
    RUN_WITH_RUNTIME_CONFIG(1, xpu_runtime_config1);
  }
  xpu_stream_destroy(stream);
}

TEST(runtime_stream, 2_new_same_stream) {
  void* stream = nullptr;
  xpu_stream_create(&stream);
  CHECK_NOTNULL(stream);
  experimental::XpuRuntimeConfig xpu_runtime_config;
  xpu_runtime_config.context = nullptr;
  xpu_runtime_config.stream = stream;
  xpu_runtime_config.l3_size = 0;
  xpu_runtime_config.l3_ptr = nullptr;
  xpu_runtime_config.l3_autotune_size = 0;
  {
    RUN_WITH_RUNTIME_CONFIG(0, xpu_runtime_config);
    RUN_WITH_RUNTIME_CONFIG(1, xpu_runtime_config);
  }
  xpu_stream_destroy(stream);
}

TEST(runtime_stream, 2_new_different_stream) {
  void* stream0 = nullptr;
  xpu_stream_create(&stream0);
  CHECK_NOTNULL(stream0);
  experimental::XpuRuntimeConfig xpu_runtime_config0;
  xpu_runtime_config0.context = nullptr;
  xpu_runtime_config0.stream = stream0;
  xpu_runtime_config0.l3_size = 0;
  xpu_runtime_config0.l3_ptr = nullptr;
  xpu_runtime_config0.l3_autotune_size = 0;
  void* stream1 = nullptr;
  xpu_stream_create(&stream1);
  CHECK_NOTNULL(stream1);
  experimental::XpuRuntimeConfig xpu_runtime_config1;
  xpu_runtime_config1.context = nullptr;
  xpu_runtime_config1.stream = stream1;
  xpu_runtime_config1.l3_size = 0;
  xpu_runtime_config1.l3_ptr = nullptr;
  xpu_runtime_config1.l3_autotune_size = 0;
  {
    RUN_WITH_RUNTIME_CONFIG(0, xpu_runtime_config0);
    RUN_WITH_RUNTIME_CONFIG(1, xpu_runtime_config1);
  }
  xpu_stream_destroy(stream0);
  xpu_stream_destroy(stream1);
}

void RunPredictorWithRuntimeConfig(
    std::shared_ptr<Predictor> predictor,
    experimental::XpuRuntimeConfig runtime_config) {
  PrepareInput(predictor);
  experimental::InternalUtils::RunWithRuntimeConfig(predictor.get(),
                                                    &runtime_config);
  CompareOutput(predictor);
  PADDLE_ENFORCE_EQ(predictor->GetExecStream(),
                    runtime_config.stream,
                    common::errors::InvalidArgument(
                        "predictor->GetExecStream() is not equal with"
                        "runtime_config.stream"));
}

TEST(runtime_stream, 2_thread) {
  void* stream0 = nullptr;
  xpu_stream_create(&stream0);
  CHECK_NOTNULL(stream0);
  experimental::XpuRuntimeConfig xpu_runtime_config0;
  xpu_runtime_config0.context = nullptr;
  xpu_runtime_config0.stream = stream0;
  xpu_runtime_config0.l3_size = 0;
  xpu_runtime_config0.l3_ptr = nullptr;
  xpu_runtime_config0.l3_autotune_size = 0;

  void* stream1 = nullptr;
  xpu_stream_create(&stream1);
  CHECK_NOTNULL(stream1);
  experimental::XpuRuntimeConfig xpu_runtime_config1;
  xpu_runtime_config1.context = nullptr;
  xpu_runtime_config1.stream = stream1;
  xpu_runtime_config1.l3_size = 0;
  xpu_runtime_config1.l3_ptr = nullptr;
  xpu_runtime_config1.l3_autotune_size = 0;

  {
    RUN_WITH_RUNTIME_CONFIG(0, xpu_runtime_config0);
    RUN_WITH_RUNTIME_CONFIG(1, xpu_runtime_config1);
    std::thread t0(
        RunPredictorWithRuntimeConfig, predictor0, xpu_runtime_config0);
    std::thread t1(
        RunPredictorWithRuntimeConfig, predictor1, xpu_runtime_config1);
    t0.join();
    t1.join();
  }

  xpu_stream_destroy(stream0);
  xpu_stream_destroy(stream1);
}

TEST(runtime_context, new_context) {
  auto* context = baidu::xpu::api::create_context();
  CHECK_NOTNULL(context);
  {
    experimental::XpuRuntimeConfig xpu_runtime_config;
    xpu_runtime_config.context = context;
    xpu_runtime_config.stream = nullptr;
    xpu_runtime_config.l3_size = 0;
    xpu_runtime_config.l3_ptr = nullptr;
    xpu_runtime_config.l3_autotune_size = 0;
    RUN_WITH_RUNTIME_CONFIG(0, xpu_runtime_config);
  }
  baidu::xpu::api::destroy_context(context);
}

}  // namespace paddle_infer
