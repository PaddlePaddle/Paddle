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
    engine_ = new TensorRTEngine(10, 1 << 10, stream_);
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
  engine_->GetOutputInCPU("y", &y_cpu, 1 * sizeof(float));

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

  auto dims = engine_->GetITensor("y")->getDimensions();
  ASSERT_EQ(dims.nbDims, 3);
  ASSERT_EQ(dims.d[0], 2);
  ASSERT_EQ(dims.d[1], 1);
  engine_->GetOutputInCPU("y", &y_cpu[0], 2 * sizeof(float));
  ASSERT_EQ(y_cpu[0], 4.5);
  ASSERT_EQ(y_cpu[1], 14.5);
}

TEST_F(TensorRTEngineTest, test_conv2d) {
  // Weight in CPU memory.
  float raw_weight[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float raw_bias[1] = {0};

  TensorRTEngine::Weight weight(nvinfer1::DataType::kFLOAT, raw_weight, 9);
  TensorRTEngine::Weight bias(nvinfer1::DataType::kFLOAT, raw_bias, 1);
  auto* x = engine_->DeclareInput("x", nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims3{1, 3, 3});
  auto* conv_layer =
      TRT_ENGINE_ADD_LAYER(engine_, Convolution, *x, 1, nvinfer1::DimsHW{3, 3},
                           weight.get(), bias.get());
  PADDLE_ENFORCE(conv_layer != nullptr);
  conv_layer->setStride(nvinfer1::DimsHW{1, 1});
  conv_layer->setPadding(nvinfer1::DimsHW{1, 1});

  engine_->DeclareOutput(conv_layer, 0, "y");
  engine_->FreezeNetwork();
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);

  float x_v[18] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  engine_->SetInputFromCPU("x", reinterpret_cast<void*>(&x_v),
                           18 * sizeof(float));
  engine_->Execute(2);

  LOG(INFO) << "to get output";
  float* y_cpu = new float[18];
  engine_->GetOutputInCPU("y", &y_cpu[0], 18 * sizeof(float));
  ASSERT_EQ(y_cpu[0], 4.0);
  ASSERT_EQ(y_cpu[1], 6.0);
}

TEST_F(TensorRTEngineTest, test_pool2d) {
  // Weight in CPU memory.
  auto* x = engine_->DeclareInput("x", nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims3{1, 2, 2});

  nvinfer1::PoolingType pool_t = nvinfer1::PoolingType::kAVERAGE;
  auto* pool_layer =
      TRT_ENGINE_ADD_LAYER(engine_, Pooling, *const_cast<nvinfer1::ITensor*>(x),
                           pool_t, nvinfer1::DimsHW{2, 2});

  PADDLE_ENFORCE(pool_layer != nullptr);
  pool_layer->setStride(nvinfer1::DimsHW{1, 1});
  pool_layer->setPadding(nvinfer1::DimsHW{0, 0});

  engine_->DeclareOutput(pool_layer, 0, "y");
  engine_->FreezeNetwork();
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);

  float x_v[8] = {1.0, 2.0, 5.0, 0.0, 2.0, 3.0, 5.0, 10.0};
  engine_->SetInputFromCPU("x", reinterpret_cast<void*>(&x_v),
                           8 * sizeof(float));
  engine_->Execute(2);

  LOG(INFO) << "to get output";
  float* y_cpu = new float[2];
  engine_->GetOutputInCPU("y", &y_cpu[0], 2 * sizeof(float));

  ASSERT_EQ(y_cpu[0], 2.0);
  ASSERT_EQ(y_cpu[1], 5.0);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
