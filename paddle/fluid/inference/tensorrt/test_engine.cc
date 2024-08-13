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
#include <memory>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::inference::tensorrt {

class TensorRTEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ctx_ = new phi::GPUContext(phi::GPUPlace(0));
    ctx_->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(phi::GPUPlace(0), ctx_->stream())
                           .get());
    ctx_->SetHostAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::CPUPlace())
            .get());
    ctx_->SetZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(phi::GPUPlace(0))
            .get());
    ctx_->SetHostZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(phi::CPUPlace())
            .get());
    ctx_->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::GPUPinnedPlace())
            .get());
    ctx_->PartialInitWithAllocator();

    TensorRTEngine::ConstructionParams params;
    params.max_batch_size = 10;
    params.max_workspace_size = 1 << 10;
    params.with_dynamic_shape = true;
    engine_ = std::make_unique<TensorRTEngine>(params);
    engine_->InitNetwork();
  }

  void PrepareInputOutput(const std::vector<float> &input,
                          std::vector<int> output_shape) {
    paddle::framework::TensorFromVector(input, *ctx_, &input_);
    output_.Resize(common::make_ddim(output_shape));
  }

  void GetOutput(std::vector<float> *output) {
    paddle::framework::TensorToVector(output_, *ctx_, output);
  }

 protected:
  phi::DenseTensor input_;
  phi::DenseTensor output_;
  std::unique_ptr<TensorRTEngine> engine_ = nullptr;
  phi::GPUContext *ctx_ = nullptr;
};

TEST_F(TensorRTEngineTest, add_layer) {
  const int size = 1;

  std::vector<float> raw_weight = {2.};  // Weight in CPU memory.
  std::vector<float> raw_bias = {3.};

  std::vector<void *> buffers(2);  // TRT binded inputs

  LOG(INFO) << "create weights";
  TensorRTEngine::Weight weight(
      nvinfer1::DataType::kFLOAT, raw_weight.data(), size);
  TensorRTEngine::Weight bias(
      nvinfer1::DataType::kFLOAT, raw_bias.data(), size);
  auto *x = engine_->DeclareInput(
      "x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{1, 1, 1});
  auto *weight_layer = TRT_ENGINE_ADD_LAYER(
      engine_, Constant, nvinfer1::Dims3{1, 1, 1}, weight.get());
  auto *bias_layer = TRT_ENGINE_ADD_LAYER(
      engine_, Constant, nvinfer1::Dims3{1, 1, 1}, bias.get());
  auto *matmul_layer =
      TRT_ENGINE_ADD_LAYER(engine_,
                           MatrixMultiply,
                           *x,
                           nvinfer1::MatrixOperation::kNONE,
                           *weight_layer->getOutput(0),
                           nvinfer1::MatrixOperation::kTRANSPOSE);
  PADDLE_ENFORCE_NOT_NULL(
      matmul_layer,
      common::errors::InvalidArgument(
          "The TRT MatrixMultiply layer cannot be null. There is something "
          "wrong with the TRT network building and layer creation."));
  auto *add_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                         ElementWise,
                                         *matmul_layer->getOutput(0),
                                         *bias_layer->getOutput(0),
                                         nvinfer1::ElementWiseOperation::kSUM);
  PADDLE_ENFORCE_NOT_NULL(
      add_layer,
      common::errors::InvalidArgument(
          "The TRT elementwise layer cannot be null. There is something wrong "
          "with the TRT network building and layer creation."));

  engine_->DeclareOutput(add_layer, 0, "y");
  LOG(INFO) << "freeze network";
  engine_->FreezeNetwork();
#if IS_TRT_VERSION_GE(8600)
  ASSERT_EQ(engine_->engine()->getNbIOTensors(), 2);
#else
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);
#endif

  // fill in real data
  std::vector<float> x_v = {1234};
  std::vector<float> y_cpu;
  PrepareInputOutput(x_v, {1});

  auto *x_v_gpu_data = input_.mutable_data<float>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  buffers[0] = reinterpret_cast<void *>(x_v_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  LOG(INFO) << "to execute";
  engine_->Execute(1, &buffers, ctx_->stream());

  LOG(INFO) << "to get output";
  GetOutput(&y_cpu);

  LOG(INFO) << "to checkout output";
  ASSERT_EQ(y_cpu[0], x_v[0] * 2 + 3);
}

TEST_F(TensorRTEngineTest, add_layer_multi_dim) {
  // Weight in CPU memory.
  // It seems tensorrt FC use col-major: [[1.0, 3.3], [1.1, 4.4]]
  // instead of row-major, which is [[1.0, 1.1], [3.3, 4.4]]
  std::vector<float> raw_weight = {1.0, 1.1, 3.3, 4.4};
  std::vector<float> raw_bias = {1.3, 2.4};
  std::vector<void *> buffers(2);  // TRT binded inputs

  TensorRTEngine::Weight weight(
      nvinfer1::DataType::kFLOAT, raw_weight.data(), 4);
  TensorRTEngine::Weight bias(nvinfer1::DataType::kFLOAT, raw_bias.data(), 2);
  auto *x = engine_->DeclareInput(
      "x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{1, 1, 2});
  auto *weight_layer = TRT_ENGINE_ADD_LAYER(
      engine_, Constant, nvinfer1::Dims3{1, 2, 2}, weight.get());
  auto *bias_layer = TRT_ENGINE_ADD_LAYER(
      engine_, Constant, nvinfer1::Dims3{1, 1, 2}, bias.get());
  auto *matmul_layer =
      TRT_ENGINE_ADD_LAYER(engine_,
                           MatrixMultiply,
                           *x,
                           nvinfer1::MatrixOperation::kNONE,
                           *weight_layer->getOutput(0),
                           nvinfer1::MatrixOperation::kTRANSPOSE);
  PADDLE_ENFORCE_NOT_NULL(
      matmul_layer,
      common::errors::InvalidArgument(
          "The TRT MatrixMultiply layer cannot be null. There is something "
          "wrong with the TRT network building and layer creation."));
  auto *add_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                         ElementWise,
                                         *matmul_layer->getOutput(0),
                                         *bias_layer->getOutput(0),
                                         nvinfer1::ElementWiseOperation::kSUM);
  PADDLE_ENFORCE_NOT_NULL(
      add_layer,
      common::errors::InvalidArgument(
          "The TRT elementwise layer cannot be null. There is something wrong "
          "with the TRT network building and layer creation."));

  engine_->DeclareOutput(add_layer, 0, "y");
  engine_->FreezeNetwork();
#if IS_TRT_VERSION_GE(8600)
  ASSERT_EQ(engine_->engine()->getNbIOTensors(), 2);
#else
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);
#endif

  // fill in real data
  std::vector<float> x_v = {1.0, 2.0};
  std::vector<float> y_cpu;
  PrepareInputOutput(x_v, {2});

  auto *x_v_gpu_data = input_.mutable_data<float>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  buffers[0] = reinterpret_cast<void *>(x_v_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  engine_->Execute(1, &buffers, ctx_->stream());

  LOG(INFO) << "to get output";
  GetOutput(&y_cpu);

  auto dims = engine_->GetITensor("y")->getDimensions();
  ASSERT_EQ(dims.nbDims, 3);
  ASSERT_EQ(dims.d[0], 1);
  ASSERT_EQ(dims.d[1], 1);
  ASSERT_EQ(dims.d[2], 2);

  ASSERT_EQ(y_cpu[0], 4.5);
  ASSERT_EQ(y_cpu[1], 14.5);
}

TEST_F(TensorRTEngineTest, test_conv2d) {
  // Weight in CPU memory.
  std::vector<float> raw_weight = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<float> raw_bias = {0};
  std::vector<void *> buffers(2);  // TRT binded inputs

  TensorRTEngine::Weight weight(
      nvinfer1::DataType::kFLOAT, raw_weight.data(), 9);
  TensorRTEngine::Weight bias(nvinfer1::DataType::kFLOAT, raw_bias.data(), 1);
  auto *x = engine_->DeclareInput(
      "x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{2, 1, 3, 3});
  auto *conv_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                          ConvolutionNd,
                                          *x,
                                          1,
                                          nvinfer1::DimsHW{3, 3},
                                          weight.get(),
                                          bias.get());
  PADDLE_ENFORCE_NOT_NULL(conv_layer,
                          common::errors::InvalidArgument(
                              "TRT convolution layer building failed."));
  conv_layer->setStrideNd(nvinfer1::Dims2{1, 1});
  conv_layer->setPaddingNd(nvinfer1::Dims2{1, 1});

  engine_->DeclareOutput(conv_layer, 0, "y");
  engine_->FreezeNetwork();
#if IS_TRT_VERSION_GE(8600)
  ASSERT_EQ(engine_->engine()->getNbIOTensors(), 2);
#else
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);
#endif

  // fill in real data
  std::vector<float> x_v = {1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0};
  std::vector<float> y_cpu;
  PrepareInputOutput(x_v, {18});

  auto *x_v_gpu_data = input_.mutable_data<float>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  buffers[0] = reinterpret_cast<void *>(x_v_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  engine_->Execute(2, &buffers, ctx_->stream());

  LOG(INFO) << "to get output";
  GetOutput(&y_cpu);

  ASSERT_EQ(y_cpu[0], 4.0);
  ASSERT_EQ(y_cpu[1], 6.0);
}

TEST_F(TensorRTEngineTest, test_pool2d) {
  // Weight in CPU memory.
  auto *x = engine_->DeclareInput(
      "x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{2, 1, 2, 2});

  std::vector<void *> buffers(2);  // TRT binded inputs
  nvinfer1::PoolingType pool_t = nvinfer1::PoolingType::kAVERAGE;
  auto *pool_layer = TRT_ENGINE_ADD_LAYER(
      engine_, PoolingNd, *x, pool_t, nvinfer1::DimsHW{2, 2});

  PADDLE_ENFORCE_NOT_NULL(
      pool_layer,
      common::errors::InvalidArgument("TRT pooling layer building failed."));
  pool_layer->setStrideNd(nvinfer1::Dims2{1, 1});
  pool_layer->setPaddingNd(nvinfer1::Dims2{0, 0});

  engine_->DeclareOutput(pool_layer, 0, "y");
  engine_->FreezeNetwork();
#if IS_TRT_VERSION_GE(8600)
  ASSERT_EQ(engine_->engine()->getNbIOTensors(), 2);
#else
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);
#endif

  // fill in real data
  std::vector<float> x_v = {1.0, 2.0, 5.0, 0.0, 2.0, 3.0, 5.0, 10.0};
  std::vector<float> y_cpu;
  PrepareInputOutput(x_v, {2});

  auto *x_v_gpu_data = input_.mutable_data<float>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  buffers[0] = reinterpret_cast<void *>(x_v_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  engine_->SetAllNodesLowerToTrt(true);
  engine_->Execute(2, &buffers, ctx_->stream());

  LOG(INFO) << "to get output";
  GetOutput(&y_cpu);

  ASSERT_EQ(y_cpu[0], 2.0);
  ASSERT_EQ(y_cpu[1], 5.0);
}

}  // namespace paddle::inference::tensorrt
