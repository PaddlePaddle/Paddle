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
#include "gflags/gflags.h"

#include "paddle/fluid/inference/tests/api/tester_helper.h"
#include "paddle/pten/api/lib/device_context_pool.h"

namespace paddle {
namespace inference {

int test_predictor(const AnalysisConfig& config_in,
                   Barrier* barrier = nullptr) {
  static std::mutex mutex;
  AnalysisConfig config{config_in};
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
      -0.00621776f, -0.00620937f, 0.00990623f,  -0.0039817f, -0.00074315f,
      0.61229795f,  -0.00491806f, -0.00068755f, 0.18409646f, 0.30090684f};
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
  config.SwitchUseFeedFetchOps(false);
  std::unique_ptr<PaddlePredictor> predictor;
  {
    std::unique_lock<std::mutex> lock(mutex);
    predictor = std::move(CreatePaddlePredictor(config));
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

  const std::vector<float> truth_values = {
      -0.00621776f, -0.00620937f, 0.00990623f,  -0.0039817f, -0.00074315f,
      0.61229795f,  -0.00491806f, -0.00068755f, 0.18409646f, 0.30090684f};
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

#ifdef LITE_SUBGRAPH_WITH_XPU
TEST(AnalysisPredictor, lite_xpu) {
  AnalysisConfig config;
  config.EnableXpu();
  config.SetModel(FLAGS_infer_model + "/" + "mul_model");
  config.EnableLiteEngine(paddle::AnalysisConfig::Precision::kFloat32);
  test_predictor(config);
  test_predictor_zero_copy(config);
}
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
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
      test_predictor(config, &barrier);
      test_predictor_zero_copy(config);
    });
  }
  for (auto& th : threads) {
    th.join();
  }
}

TEST(AnalysisPredictor_norm, lite_engine) {
  auto place = paddle::platform::CUDAPlace();
  auto* fluid_ctx = static_cast<platform::CUDADeviceContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  VLOG(3) << "The prediction process will be completed using a separate "
             "normal-priority stream on each thread.";
  fluid_ctx->ResetThreadContext(platform::stream::Priority::kNormal);

  // TODO(wilber): seems temporarily lose thread_stream ability, need to fix
  // after pten is done.
  auto* pten_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto* ptr = dynamic_cast<pten::CUDAContext*>(pten_ctx);
  ptr->SetCUDAMaxGridDimX(fluid_ctx->GetCUDAMaxGridDimSize().x);
  ptr->SetCUDAMaxGridDimY(fluid_ctx->GetCUDAMaxGridDimSize().y);
  ptr->SetCUDAMaxGridDimZ(fluid_ctx->GetCUDAMaxGridDimSize().z);
  ptr->SetSMCount(fluid_ctx->GetSMCount());
  ptr->SetTensorCoreAvailable(fluid_ctx->tensor_core_available());
  ptr->SetComputeCapability(fluid_ctx->GetComputeCapability());
  ptr->SetMaxThreadsPerBlock(fluid_ctx->GetMaxThreadsPerBlock());

  // need to set 3 cublas handle?
  ptr->SetCublasHandle(fluid_ctx->cublas_handle());
  //  device_ctx->cublas_handle()

  // Fluid now only support one stream.
  ptr->SetStream(fluid_ctx->stream());
  ptr->SetHostToDeviceStream(fluid_ctx->stream());
  ptr->SetDeviceToHostStream(fluid_ctx->stream());

#ifdef PADDLE_WITH_CUDNN
  ptr->SetCudnnHandle(fluid_ctx->cudnn_handle());
#endif

#ifdef PADDLE_WITH_NCCL
  ptr->SetNcclComm(fluid_ctx->nccl_comm());
#endif

  // #if defined(PADDLE_WITH_EIGEN) && !defined(PADDLE_WITH_HIP)
  ptr->SetEigenDevice(fluid_ctx->eigen_device());
  // #endif

  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel(FLAGS_infer_model + "/" + "mul_model");
  config.EnableLiteEngine(paddle::AnalysisConfig::Precision::kFloat32);
  test_predictor(config);
  test_predictor_zero_copy(config);
}
#endif

}  // namespace inference
}  // namespace paddle
