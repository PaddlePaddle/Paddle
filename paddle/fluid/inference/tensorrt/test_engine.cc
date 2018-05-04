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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class TensorRTEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, cudaStreamCreate(&stream_));
    engine_ = new TensorRTEngine(1, 1 << 10, &stream_);
    engine_->InitNetwork();
  }

  void TearDown() override {
    delete engine_;
    cudaStreamDestroy(stream_);
  }

 protected:
  TensorRTEngine* engine_;
  cudaStream_t stream_;
};

TEST_F(TensorRTEngineTest, add_layer) {
  const int size = 1;

  float raw_weight[size] = {2.};  // Weight in CPU memory.
  float raw_bias[size] = {3.};

  LOG(INFO) << "create weights";
  TensorRTEngine::Weight weight(nvinfer1::DataType::kFLOAT, raw_weight, size);
  TensorRTEngine::Weight bias(nvinfer1::DataType::kFLOAT, raw_bias, size);
  auto* x = engine_->DeclareInput("x", nvinfer1::DataType::kFLOAT,
                                  nvinfer1::DimsCHW{1, 1, 1});
  auto* fc_layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *x, size,
                                        weight.get(), bias.get());
  PADDLE_ENFORCE(fc_layer != nullptr);

  engine_->DeclareOutput(fc_layer, 0, "y");
  LOG(INFO) << "freeze network";
  engine_->FreezeNetwork();
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);

  // fill in real data
  float x_v = 1234;
  engine_->SetInputFromCPU("x", reinterpret_cast<void*>(&x_v),
                           1 * sizeof(float));
  LOG(INFO) << "to execute";
  engine_->Execute(1);

  LOG(INFO) << "to get output";
  float y_cpu;
  engine_->GetOutputInCPU("y", &y_cpu, sizeof(float));

  LOG(INFO) << "to checkout output";
  ASSERT_EQ(y_cpu, x_v * 2 + 3);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
