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

#include "framework/core/net/net.h"
#include "framework/graph/graph.h"
#include "paddle/fluid/inference/anakin/engine.h"

using anakin::Precision;
using anakin::saber::NV;
using anakin::saber::X86;
namespace paddle {
namespace inference {
namespace anakin {

class TestAnakinEngine : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override {}

 protected:
  using AnakinNvEngineT = AnakinEngine<NV, Precision::FP32>;
  std::unique_ptr<AnakinNvEngineT> engine_{nullptr};
};

void TestAnakinEngine::SetUp() {
  engine_.reset(new AnakinEngine<NV, Precision::FP32>(true));
  engine_->DeclareInputs({"x"});
  engine_->DeclareOutputs({"y"});
}

TEST_F(TestAnakinEngine, Execute) {
  engine_->AddOp("op1", "Dense", {"x"}, {"y"});
  engine_->AddOpAttr("op1", "out_dim", 100);
  engine_->AddOpAttr("op1", "bias_term", false);
  engine_->AddOpAttr("op1", "axis", 2);
  std::vector<int> shape = {1, 1, 3, 100};
  ::anakin::saber::Shape tmp_shape(shape);
  //::anakin::saber::Tensor<::anakin::saber::NV> weight1(tmp_shape);
  ::anakin::PBlock<::anakin::saber::NV> weight1(tmp_shape);
  engine_->AddOpAttr("op1", "weight_1", weight1);
  // engine_->AddOpAttr("op1", "Inputs");
  ::anakin::PTuple<int> input_shape = {1, 1, 1, 3};
  // engine_->AddOpAttr<::anakin::PTuple<int>>("x", "input_shape", {1, 1, 1,
  // 3});
  engine_->FreezeNetwork();
  engine_->AddOpAttr("x", "input_shape", input_shape);
  engine_->Init();

}

/*
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
*/

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
