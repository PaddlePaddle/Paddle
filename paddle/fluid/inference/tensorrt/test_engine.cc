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

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/spmm_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class TensorRTEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ctx_ = new platform::CUDADeviceContext(platform::CUDAPlace(0));
    ctx_->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(platform::CUDAPlace(0), ctx_->stream())
                           .get());
    ctx_->SetHostAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(paddle::platform::CPUPlace())
            .get());
    ctx_->SetZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(platform::CUDAPlace(0))
            .get());
    ctx_->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(paddle::platform::CUDAPinnedPlace())
            .get());
    ctx_->PartialInitWithAllocator();

    engine_ = new TensorRTEngine(16, 1 << 10);
    engine_->InitNetwork();
  }

  void TearDown() override {
    if (engine_) {
      delete engine_;
      engine_ = nullptr;
    }
  }

  void PrepareInputOutput(const std::vector<float> &input,
                          std::vector<int> output_shape) {
    paddle::framework::TensorFromVector(input, *ctx_, &input_);
    output_.Resize(phi::make_ddim(output_shape));
  }

  void GetOutput(std::vector<float> *output) {
    paddle::framework::TensorToVector(output_, *ctx_, output);
  }

 protected:
  framework::Tensor input_;
  framework::Tensor output_;
  TensorRTEngine *engine_;
  platform::CUDADeviceContext *ctx_;
};

TEST_F(TensorRTEngineTest, test_sparse_fc) {
  // Weight in CPU memory.
  // float raw_weight[16] = {1.0, 0, 0, 4.4,
  //                        1.0, 0, 0, 4.4,
  //                        1.0, 0, 0, 4.4,
  //                        1.0, 0, 0, 4.4};
  float raw_weight[256];
  for (int i=0; i<64; i++) {
    raw_weight[4*i] = 1.0;
    raw_weight[4*i+1] = 0.0;
    raw_weight[4*i+2] = 0.0;
    raw_weight[4*i+3] = 4.4;
  }
  float raw_bias[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<void *> buffers(2);  // TRT binded inputs
  // builder->setMaxBatchSize(1)
  TensorRTEngine::Weight weight(nvinfer1::DataType::kFLOAT, raw_weight, 256);
  TensorRTEngine::Weight bias(nvinfer1::DataType::kFLOAT, raw_bias, 16);
  auto *x = engine_->DeclareInput("x", nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims4{16, 1, 16, 1});

  plugin::SpmmPluginDynamic::Activation act = plugin::SpmmPluginDynamic::Activation::kNone;

  plugin::SpmmPluginDynamic* plugin = new plugin::SpmmPluginDynamic("CustomSpmmPluginDynamic", nvinfer1::DataType::kFLOAT, 16, weight.get(), bias.get(), act);
  std::vector<nvinfer1::ITensor*> plugin_inputs;
  plugin_inputs.emplace_back(x);
  auto fc_layer = engine_->network()->addPluginV2(
      plugin_inputs.data(), plugin_inputs.size(), *plugin);

  PADDLE_ENFORCE_NOT_NULL(fc_layer,
                          platform::errors::InvalidArgument(
                              "TRT SPARSE FC layer building failed."));

  engine_->DeclareOutput(fc_layer, 0, "y");
  engine_->FreezeNetwork();
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);

  // fill in real data
  // std::vector<float> x_v = {1.0, 2.0, 3.0, 4.0,
  //                           1.0, 2.0, 3.0, 4.0,
  //                           1.0, 2.0, 3.0, 4.0,
  //                           1.0, 2.0, 3.0, 4.0};
  std::vector<float> x_v(256);
  for (int i=0; i<64; i++) {
    x_v[4*i] = 1.0;
    x_v[4*i+1] = 2.0;
    x_v[4*i+2] = 3.0;
    x_v[4*i+3] = 4.0;
  }

  std::vector<float> y_cpu;
  PrepareInputOutput(x_v, {16});

  auto *x_v_gpu_data = input_.mutable_data<float>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  buffers[0] = reinterpret_cast<void *>(x_v_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  engine_->Execute(16, &buffers, ctx_->stream());

  LOG(INFO) << "to get output";
  GetOutput(&y_cpu);

  auto dims = engine_->GetITensor("y")->getDimensions();
  ASSERT_EQ(dims.nbDims, 2);
  ASSERT_EQ(dims.d[0], 16);

  ASSERT_EQ(y_cpu[0], 18.6);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
