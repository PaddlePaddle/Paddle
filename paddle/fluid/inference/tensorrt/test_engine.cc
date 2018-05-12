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
#include <memory>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class TensorRTEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, cudaStreamCreate(&stream_));
    engine_ = new TensorRTEngine(20, 1 << 10, &stream_);
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

TEST_F(TensorRTEngineTest, add_layer_multi_dim) {
  // Weight in CPU memory.
  // It seems tensorrt FC use col-major: [[1.0, 3.3], [1.1, 4.4]]
  // instead of row-major, which is [[1.0, 1.1], [3.3, 4.4]]
  float raw_weight[4] = {1.0, 1.1, 3.3, 4.4};
  float raw_bias[2] = {1.3, 2.4};

  TensorRTEngine::Weight weight(nvinfer1::DataType::kFLOAT, raw_weight, 4);
  TensorRTEngine::Weight bias(nvinfer1::DataType::kFLOAT, raw_bias, 2);
  auto* x = engine_->DeclareInput("x", nvinfer1::DataType::kFLOAT,
                                  nvinfer1::DimsCHW{1, 2, 1});
  auto* fc_layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *x, 2,
                                        weight.get(), bias.get());
  PADDLE_ENFORCE(fc_layer != nullptr);

  engine_->DeclareOutput(fc_layer, 0, "y");
  engine_->FreezeNetwork();
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);

  float x_v[2] = {1.0, 2.0};
  engine_->SetInputFromCPU("x", reinterpret_cast<void*>(&x_v),
                           2 * sizeof(float));
  engine_->Execute(1);

  LOG(INFO) << "to get output";
  float y_cpu[2] = {-1., -1.};
  engine_->GetOutputInCPU("y", &y_cpu[0], sizeof(float) * 2);
  ASSERT_EQ(y_cpu[0], 4.5);
  ASSERT_EQ(y_cpu[1], 14.5);
}

TEST_F(TensorRTEngineTest, load_onnx_official_mnist_model) {
  // fix latter.
  std::string fit_a_line_model = "/chunwei/build_dir/Paddle/mnist.onnx";
  // std::string fit_a_line_model = "/chunwei/build_dir/Paddle/build/1.onnx";
  engine_->BuildFromONNX(fit_a_line_model);
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);
  const std::string input_name = "Input3";
  const std::string output_name = "Plus214_Output_0";
  auto& ibuffer = engine_->buffer(input_name);
  auto& obuffer = engine_->buffer(output_name);
  const size_t input_dim = 784;
  const size_t output_dim = 10;
  ASSERT_EQ(ibuffer.size, input_dim);
  ASSERT_EQ(obuffer.size, output_dim);
  ASSERT_TRUE(ibuffer.device == DeviceType::GPU);
  ASSERT_TRUE(obuffer.device == DeviceType::GPU);

  const size_t batch_size = 10;
  std::unique_ptr<float> idata(new float[input_dim * batch_size]);
  std::unique_ptr<float> odata(new float[output_dim * batch_size]);

  for (size_t i = 0; i < input_dim * batch_size; i++) {
    *(idata.get() + i) = i;
  }
  cudaMemcpyAsync((void*)ibuffer.buffer, (void*)(idata.get()),
                  input_dim * batch_size * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);
  engine_->Execute(batch_size);
  cudaMemcpyAsync((void*)(odata.get()), (void*)obuffer.buffer,
                  output_dim * sizeof(float), cudaMemcpyDeviceToHost, stream_);

  for (size_t i = 0; i < batch_size; i++) {
    std::stringstream ss;
    for (int j = 0; j < 10; j++) {
      ss << *(i * output_dim + odata.get() + j) << " ";
    }
    LOG(INFO) << ss.str();
  }
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
