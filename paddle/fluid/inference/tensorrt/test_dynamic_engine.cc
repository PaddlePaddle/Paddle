/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/phi/common/data_type.h"
#if PADDLE_WITH_CUSPARSELT && IS_TRT_VERSION_GE(8000)
#include "paddle/fluid/inference/tensorrt/plugin/spmm_plugin.h"
#endif
#include "paddle/fluid/inference/tensorrt/plugin/fused_token_prune_op_plugin.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/float16.h"

using float16 = phi::dtype::float16;
namespace paddle {
namespace inference {
namespace tensorrt {

class TensorRTDynamicEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ctx_ = new phi::GPUContext(platform::CUDAPlace(0));
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

    std::map<std::string, std::vector<int>> min_input_shape = {
        {"input", {16, 32, 1, 1}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"input", {16, 32, 1, 1}}};
    std::map<std::string, std::vector<int>> optim_input_shape = {
        {"input", {16, 32, 1, 1}}};

    engine_ = new TensorRTEngine(16,
                                 1 << 10,
                                 AnalysisConfig::Precision::kHalf,
                                 nullptr,
                                 0,
                                 min_input_shape,
                                 max_input_shape,
                                 optim_input_shape,
                                 false,
                                 phi::DataType::FLOAT32,
                                 NaiveLogger::Global());
    engine_->InitNetwork();
  }

  void TearDown() override {
    if (engine_) {
      delete engine_;
      engine_ = nullptr;
    }
  }

  void PrepareInputOutput(const std::vector<float16> &input,
                          std::vector<int> output_shape) {
    paddle::framework::TensorFromVector(input, *ctx_, &input_);
    output_.Resize(phi::make_ddim(output_shape));
  }

  void GetOutput(std::vector<float> *output) {
    paddle::framework::TensorToVector(output_, *ctx_, output);
  }

 protected:
  phi::DenseTensor input_;
  phi::DenseTensor output_;
  TensorRTEngine *engine_;
  phi::GPUContext *ctx_;
};

TEST_F(TensorRTDynamicEngineTest, test_spmm) {
  // Weight in CPU memory.
#if PADDLE_WITH_CUSPARSELT && IS_TRT_VERSION_GE(8000)
  float16 raw_weight[512];
  for (int i = 0; i < 128; i++) {
    if (i % 16 <= 7) {
      raw_weight[4 * i] = float16(1.0);
      raw_weight[4 * i + 1] = float16(0.0);
      raw_weight[4 * i + 2] = float16(0.0);
      raw_weight[4 * i + 3] = float16(4.0);
    } else {
      raw_weight[4 * i] = float16(0.0);
      raw_weight[4 * i + 1] = float16(2.0);
      raw_weight[4 * i + 2] = float16(3.0);
      raw_weight[4 * i + 3] = float16(0.0);
    }
  }
  float16 raw_bias[16] = {float16(0),
                          float16(1),
                          float16(0),
                          float16(2),
                          float16(0),
                          float16(3),
                          float16(0),
                          float16(4),
                          float16(0),
                          float16(5),
                          float16(0),
                          float16(6),
                          float16(0),
                          float16(7),
                          float16(0),
                          float16(8)};
  std::vector<void *> buffers(2);  // TRT binded inputs
  TensorRTEngine::Weight weight(nvinfer1::DataType::kHALF, raw_weight, 512);
  TensorRTEngine::Weight bias(nvinfer1::DataType::kHALF, raw_bias, 16);
  std::cout << "with_dynamic_shape: " << engine_->with_dynamic_shape()
            << std::endl;
  auto *x = engine_->DeclareInput(
      "input", nvinfer1::DataType::kHALF, nvinfer1::Dims4{-1, 32, 1, 1});

  plugin::SpmmPluginDynamic::Activation act =
      plugin::SpmmPluginDynamic::Activation::kNone;

  plugin::SpmmPluginDynamic *plugin =
      new plugin::SpmmPluginDynamic("CustomSpmmPluginDynamic",
                                    nvinfer1::DataType::kHALF,
                                    16,
                                    weight.get(),
                                    bias.get(),
                                    act);
  std::vector<nvinfer1::ITensor *> plugin_inputs;
  plugin_inputs.emplace_back(x);
  auto fc_layer = engine_->network()->addPluginV2(
      plugin_inputs.data(), plugin_inputs.size(), *plugin);

  LOG(INFO) << "create weights";
  PADDLE_ENFORCE_NOT_NULL(
      fc_layer,
      platform::errors::InvalidArgument("TRT SPMM layer building failed."));

  engine_->DeclareOutput(fc_layer, 0, "y");
  engine_->FreezeNetwork();
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);

  std::vector<float16> x_v(512);
  for (int i = 0; i < 128; i++) {
    x_v[4 * i] = float16(1.0);
    x_v[4 * i + 1] = float16(2.0);
    x_v[4 * i + 2] = float16(3.0);
    x_v[4 * i + 3] = float16(4.0);
  }

  std::vector<float> y_cpu;
  PrepareInputOutput(x_v, {16, 16});

  auto *x_v_gpu_data = input_.mutable_data<float16>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  buffers[0] = reinterpret_cast<void *>(x_v_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  engine_->Execute(16, &buffers, ctx_->stream());
  LOG(INFO) << "to get output";
  GetOutput(&y_cpu);

  auto dims = engine_->GetITensor("y")->getDimensions();
  ASSERT_EQ(dims.nbDims, 4);
  ASSERT_EQ(dims.d[1], 16);
  ASSERT_EQ(y_cpu[0], 136);

  ASSERT_EQ(y_cpu[1], 105);
  ASSERT_EQ(y_cpu[32], 136);
  ASSERT_EQ(y_cpu[64], 136);
  ASSERT_EQ(y_cpu[96], 136);
#endif
  return;
}

class TensorRTDynamicTestFusedTokenPrune : public ::testing::Test {
 protected:
  void SetUp() override {
    ctx_ = new phi::GPUContext(platform::CUDAPlace(0));
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

    std::map<std::string, std::vector<int>> min_input_shape = {
        {"attn", {4, 1, 4, 4}},
        {"x", {4, 4, 1}},
        {"mask", {4, 1, 4, 4}},
        {"new_mask", {4, 1, 2, 2}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"attn", {4, 1, 4, 4}},
        {"x", {4, 4, 1}},
        {"mask", {4, 1, 4, 4}},
        {"new_mask", {4, 1, 2, 2}}};
    std::map<std::string, std::vector<int>> optim_input_shape = {
        {"attn", {4, 1, 4, 4}},
        {"x", {4, 4, 1}},
        {"mask", {4, 1, 4, 4}},
        {"new_mask", {4, 1, 2, 2}}};

    engine_ = new TensorRTEngine(16,
                                 1 << 10,
                                 AnalysisConfig::Precision::kHalf,
                                 nullptr,
                                 0,
                                 min_input_shape,
                                 max_input_shape,
                                 optim_input_shape,
                                 false,
                                 phi::DataType::FLOAT32,
                                 NaiveLogger::Global());
    engine_->InitNetwork();
  }

  void TearDown() override {
    if (engine_) {
      delete engine_;
      engine_ = nullptr;
    }
  }

  void PrepareInputOutput(const std::vector<std::vector<float16>> inputs,
                          std::vector<std::vector<int>> output_shapes) {
    LOG(INFO) << "PrepareInputOutput";
    int num_inputs = inputs.size();
    int num_outputs = output_shapes.size();
    inputs_.resize(num_inputs);
    outputs_.resize(num_outputs);
    for (int i = 0; i < num_inputs; ++i) {
      paddle::framework::TensorFromVector(inputs[i], *ctx_, &inputs_[i]);
    }
    for (int i = 0; i < num_outputs; ++i) {
      outputs_[i].Resize(phi::make_ddim(output_shapes[i]));
    }
  }

  void GetOutput(std::vector<float> &slimmed_x,     // NOLINT
                 std::vector<int32_t> &cls_inds) {  // NOLINT
    paddle::framework::TensorToVector(outputs_[0], *ctx_, &slimmed_x);
    paddle::framework::TensorToVector(outputs_[1], *ctx_, &cls_inds);
  }

 protected:
  std::vector<phi::DenseTensor> inputs_;
  std::vector<phi::DenseTensor> outputs_;
  TensorRTEngine *engine_;
  phi::GPUContext *ctx_;
};

TEST_F(TensorRTDynamicTestFusedTokenPrune, test_fused_token_prune) {
#if IS_TRT_VERSION_GE(8000)
  tensorrt::plugin::TrtPluginRegistry::Global()->RegistToTrt();
  auto *attn = engine_->DeclareInput(
      "attn", nvinfer1::DataType::kHALF, nvinfer1::Dims4{-1, 1, 4, 4});
  auto *x = engine_->DeclareInput(
      "x", nvinfer1::DataType::kHALF, nvinfer1::Dims3{-1, 4, 1});
  auto *mask = engine_->DeclareInput(
      "mask", nvinfer1::DataType::kHALF, nvinfer1::Dims4{-1, 1, 4, 4});
  auto *new_mask = engine_->DeclareInput(
      "new_mask", nvinfer1::DataType::kHALF, nvinfer1::Dims4{-1, 1, 2, 2});
  plugin::FusedTokenPrunePluginDynamic *plugin =
      new plugin::FusedTokenPrunePluginDynamic(true,
                                               /*keep_first_token*/ false,
                                               /*keep_order*/ true,
                                               /*flag_varseqlen*/ false);
  std::vector<nvinfer1::ITensor *> itensors = {attn, x, mask, new_mask};
  auto *layer = engine_->AddDynamicPlugin(itensors.data(), 4, plugin);
  PADDLE_ENFORCE_NOT_NULL(layer,
                          platform::errors::InvalidArgument(
                              "TRT fused_token_prune layer building failed."));
  std::vector<std::string> output_tensor_names{"out_slimmed_x", "out_cls_inds"};
  for (size_t i = 0; i < 2; i++) {
    layer->getOutput(i)->setName(output_tensor_names[i].c_str());
    engine_->DeclareOutput(layer, i, output_tensor_names[i]);
  }
  engine_->FreezeNetwork();

  ASSERT_EQ(engine_->engine()->getNbBindings(), 6);
  LOG(INFO) << "create input";
  std::vector<float16> attn_v(64);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        attn_v[i * 16 + j * 4 + k] = k;
      }
    }
  }
  std::vector<float16> x_v(16);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      x_v[i * 4 + j] = 1;
    }
  }
  std::vector<float16> mask_v(64);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        mask_v[i * 16 + j * 4 + k] = 1;
      }
    }
  }
  std::vector<float16> new_mask_v(16);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        new_mask_v[i * 4 + j * 2 + k] = 1;
      }
    }
  }

  LOG(INFO) << "create output";
  std::vector<int> out_slimmed_x_shape{4, 2, 1};
  std::vector<int> out_cls_ins_shape{4, 2};

  PrepareInputOutput({attn_v, x_v, mask_v, new_mask_v},
                     {out_slimmed_x_shape, out_cls_ins_shape});

  auto *attn_gpu_data = inputs_[0].mutable_data<float16>(ctx_->GetPlace());
  auto *x_gpu_data = inputs_[1].mutable_data<float16>(ctx_->GetPlace());
  auto *mask_gpu_data = inputs_[2].mutable_data<float16>(ctx_->GetPlace());
  auto *new_mask_gpu_data = inputs_[3].mutable_data<float16>(ctx_->GetPlace());

  auto *slimmed_x_gpu_data = outputs_[0].mutable_data<float>(ctx_->GetPlace());
  auto *cls_inds_gpu_data = outputs_[1].mutable_data<int32_t>(ctx_->GetPlace());

  LOG(INFO) << "create buffers";

  std::vector<void *> buffers(6);
  buffers[0] = reinterpret_cast<void *>(attn_gpu_data);
  buffers[1] = reinterpret_cast<void *>(x_gpu_data);
  buffers[2] = reinterpret_cast<void *>(mask_gpu_data);
  buffers[3] = reinterpret_cast<void *>(new_mask_gpu_data);
  buffers[4] = reinterpret_cast<void *>(slimmed_x_gpu_data);
  buffers[5] = reinterpret_cast<void *>(cls_inds_gpu_data);

  LOG(INFO) << "Execute";

  engine_->Execute(4, &buffers, ctx_->stream());

  std::vector<float> slimmed_x_v;
  std::vector<int32_t> cls_inds_v;

  LOG(INFO) << "GetOutput";
  GetOutput(slimmed_x_v, cls_inds_v);

  ASSERT_EQ(cls_inds_v[0], 2);
  ASSERT_EQ(cls_inds_v[1], 3);
  ASSERT_EQ(cls_inds_v[2], 2);
  ASSERT_EQ(cls_inds_v[3], 3);
  ASSERT_EQ(cls_inds_v[4], 2);
  ASSERT_EQ(cls_inds_v[5], 3);
  ASSERT_EQ(cls_inds_v[6], 2);
  ASSERT_EQ(cls_inds_v[7], 3);
  LOG(INFO) << "finish";
#endif
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
