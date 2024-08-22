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

#include "paddle/common/layout.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/phi/common/data_type.h"
#if PADDLE_WITH_CUSPARSELT && IS_TRT_VERSION_GE(8000)
#include "paddle/fluid/inference/tensorrt/plugin/spmm_plugin.h"
#endif
#include "paddle/fluid/inference/tensorrt/plugin/fused_token_prune_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/group_norm_op_plugin.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/float16.h"

using float16 = phi::dtype::float16;
namespace paddle::inference::tensorrt {

class TensorRTDynamicShapeValueEngineTest : public ::testing::Test {
 public:
  TensorRTDynamicShapeValueEngineTest() : engine_(nullptr), ctx_(nullptr) {}

 protected:
  void SetUp() override {
    ctx_ = std::make_unique<phi::GPUContext>(phi::GPUPlace(0));
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
    ctx_->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::GPUPinnedPlace())
            .get());
    ctx_->PartialInitWithAllocator();

    std::map<std::string, std::vector<int>> min_input_shape = {
        {"input", {1, 32}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"input", {18, 32}}};
    std::map<std::string, std::vector<int>> optim_input_shape = {
        {"input", {18, 32}}};
    std::map<std::string, std::vector<int>> min_input_value = {
        {"shape", {1, 8, 4}}};
    std::map<std::string, std::vector<int>> max_input_value = {
        {"shape", {18, 8, 4}}};
    std::map<std::string, std::vector<int>> optim_input_value = {
        {"shape", {18, 8, 4}}};

    TensorRTEngine::ConstructionParams params;
    params.max_batch_size = 16;
    params.max_workspace_size = 1 << 10;
    params.with_dynamic_shape = true;
    params.min_input_shape = min_input_shape;
    params.max_input_shape = max_input_shape;
    params.optim_input_shape = optim_input_shape;
    params.min_shape_tensor = min_input_value;
    params.max_shape_tensor = max_input_value;
    params.optim_shape_tensor = optim_input_value;

    engine_ = std::make_unique<TensorRTEngine>(params, NaiveLogger::Global());

    engine_->InitNetwork();
  }

  void PrepareInputOutput(const std::vector<float> &input,
                          std::vector<int> output_shape) {
    paddle::framework::TensorFromVector(input, *ctx_, &input_);
    output_.Resize(common::make_ddim(output_shape));
  }
  void PrepareShapeInput(const std::vector<int> &input) {
    paddle::framework::TensorFromVector(input, *ctx_, &shape_);
  }
  void GetOutput(std::vector<float> *output) {
    paddle::framework::TensorToVector(output_, *ctx_, output);
  }

 protected:
  phi::DenseTensor input_;
  phi::DenseTensor shape_;
  phi::DenseTensor output_;
  std::unique_ptr<TensorRTEngine> engine_;
  std::unique_ptr<phi::GPUContext> ctx_;
};

TEST_F(TensorRTDynamicShapeValueEngineTest, test_trt_dynamic_shape_value) {
  std::vector<void *> buffers(3);
  std::cout << "with_dynamic_shape: " << engine_->with_dynamic_shape()
            << std::endl;
  auto *x = engine_->DeclareInput(
      "input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, 32});
  nvinfer1::Dims shape_dim;
  shape_dim.nbDims = 1;
  shape_dim.d[0] = 3;
  auto *shape =
      engine_->DeclareInput("shape", nvinfer1::DataType::kINT32, shape_dim);
  auto layer = engine_->network()->addShuffle(*x);
  layer->setInput(1, *shape);
  PADDLE_ENFORCE_NOT_NULL(
      layer,
      common::errors::InvalidArgument("TRT shuffle layer building failed."));
  engine_->DeclareOutput(layer, 0, "y");
  engine_->FreezeNetwork();
#if IS_TRT_VERSION_GE(8600)
  ASSERT_EQ(engine_->engine()->getNbIOTensors(), 3);
#else
  ASSERT_EQ(engine_->engine()->getNbBindings(), 3);
#endif

  std::vector<float> x_v(8 * 32);
  for (int i = 0; i < 8 * 32; i++) {
    x_v[i] = i % (8 * 32);
  }

  std::vector<int> shape_v = {8, 8, 4};
  PrepareInputOutput(x_v, {8, 8, 4});
  PrepareShapeInput(shape_v);
#if IS_TRT_VERSION_GE(8500)
  engine_->context()->setInputShape("input", nvinfer1::Dims2{8, 32});
#else
  engine_->context()->setBindingDimensions(0, nvinfer1::Dims2{8, 32});
  engine_->context()->setBindingDimensions(1, shape_dim);
  engine_->context()->setInputShapeBinding(1, shape_v.data());
#endif
  auto *x_gpu_data = input_.mutable_data<float>(ctx_->GetPlace());
  auto *shape_gpu_data = shape_.mutable_data<int>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  buffers[0] = reinterpret_cast<void *>(x_gpu_data);
  buffers[1] = reinterpret_cast<void *>(shape_gpu_data);
  buffers[2] = reinterpret_cast<void *>(y_gpu_data);
#if IS_TRT_VERSION_GE(8500)
  for (size_t i = 0; i < buffers.size(); i++) {
    auto name = engine_->engine()->getIOTensorName(i);
    if (engine_->engine()->isShapeInferenceIO(name) &&
        engine_->engine()->getTensorIOMode(name) ==
            nvinfer1::TensorIOMode::kINPUT) {
      engine_->context()->setTensorAddress(name, shape_v.data());
    } else {
      engine_->context()->setTensorAddress(name, buffers[i]);
    }
  }
#endif

  engine_->Execute(-1, &buffers, ctx_->stream());
  cudaStreamSynchronize(ctx_->stream());

  std::vector<float> y_cpu;
  GetOutput(&y_cpu);
  ASSERT_EQ(y_cpu[0], 0);
  ASSERT_EQ(y_cpu[1], 1);
#if IS_TRT_VERSION_GE(8500)
  const char *name1 = engine_->engine()->getIOTensorName(2);
  auto dims = engine_->context()->getTensorShape(name1);
#else
  auto dims = engine_->context()->getBindingDimensions(2);
#endif
  ASSERT_EQ(dims.nbDims, 3);
  ASSERT_EQ(dims.d[0], 8);
  ASSERT_EQ(dims.d[1], 8);
  ASSERT_EQ(dims.d[2], 4);
  return;
}

class TensorRTDynamicEngineTest : public ::testing::Test {
 protected:
  TensorRTDynamicEngineTest() : engine_(nullptr), ctx_(nullptr) {}
  void SetUp() override {
    ctx_ = std::make_unique<phi::GPUContext>(phi::GPUPlace(0));
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
    ctx_->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::GPUPinnedPlace())
            .get());
    ctx_->PartialInitWithAllocator();

    std::map<std::string, std::vector<int>> min_input_shape = {
        {"input", {16, 32, 1, 1}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"input", {16, 32, 1, 1}}};
    std::map<std::string, std::vector<int>> optim_input_shape = {
        {"input", {16, 32, 1, 1}}};

    TensorRTEngine::ConstructionParams params;
    params.max_batch_size = 16;
    params.max_workspace_size = 1 << 10;
    params.with_dynamic_shape = true;
    params.precision = phi::DataType::FLOAT16;
    params.min_input_shape = min_input_shape;
    params.max_input_shape = max_input_shape;
    params.optim_input_shape = optim_input_shape;

    engine_ = std::make_unique<TensorRTEngine>(params, NaiveLogger::Global());

    engine_->InitNetwork();
  }

  void PrepareInputOutput(const std::vector<float16> &input,
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
  std::unique_ptr<TensorRTEngine> engine_;
  std::unique_ptr<phi::GPUContext> ctx_;
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
      common::errors::InvalidArgument("TRT SPMM layer building failed."));

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
  TensorRTDynamicTestFusedTokenPrune()
      : inputs_(), outputs_(), engine_(nullptr), ctx_(nullptr) {}
  void SetUp() override {
    ctx_ = std::make_unique<phi::GPUContext>(phi::GPUPlace(0));
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
    ctx_->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::GPUPinnedPlace())
            .get());
    ctx_->PartialInitWithAllocator();

    std::map<std::string, std::vector<int>> min_input_shape = {
        {"attn", {4, 4}},
        {"x", {4, 4, 1}},
        {"mask", {4, 1, 4, 4}},
        {"new_mask", {4, 1, 2, 2}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"attn", {4, 4}},
        {"x", {4, 4, 1}},
        {"mask", {4, 1, 4, 4}},
        {"new_mask", {4, 1, 2, 2}}};
    std::map<std::string, std::vector<int>> optim_input_shape = {
        {"attn", {4, 4}},
        {"x", {4, 4, 1}},
        {"mask", {4, 1, 4, 4}},
        {"new_mask", {4, 1, 2, 2}}};

    TensorRTEngine::ConstructionParams params;
    params.max_batch_size = 16;
    params.max_workspace_size = 1 << 10;
    params.precision = phi::DataType::FLOAT32;
    params.with_dynamic_shape = true;
    params.min_input_shape = min_input_shape;
    params.max_input_shape = max_input_shape;
    params.optim_input_shape = optim_input_shape;

    engine_ = std::make_unique<TensorRTEngine>(params, NaiveLogger::Global());

    engine_->InitNetwork();
  }

  void PrepareInputOutput(const std::vector<std::vector<float>> inputs,
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
      outputs_[i].Resize(common::make_ddim(output_shapes[i]));
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
  std::unique_ptr<TensorRTEngine> engine_;
  std::unique_ptr<phi::GPUContext> ctx_;
};

TEST_F(TensorRTDynamicTestFusedTokenPrune, test_fused_token_prune) {
#if IS_TRT_VERSION_GE(8000)
  auto *attn = engine_->DeclareInput(
      "attn", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, 4});
  auto *x = engine_->DeclareInput(
      "x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{-1, 4, 1});
  auto *mask = engine_->DeclareInput(
      "mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{-1, 1, 4, 4});
  auto *new_mask = engine_->DeclareInput(
      "new_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{-1, 1, 2, 2});
  plugin::FusedTokenPrunePluginDynamic *plugin =
      new plugin::FusedTokenPrunePluginDynamic(/*with_fp16*/ false,
                                               /*keep_first_token*/ false,
                                               /*keep_order*/ true,
                                               /*flag_varseqlen*/ false);
  std::vector<nvinfer1::ITensor *> itensors = {attn, x, mask, new_mask};
  auto *layer = engine_->AddDynamicPlugin(itensors.data(), 4, plugin);
  PADDLE_ENFORCE_NOT_NULL(layer,
                          common::errors::InvalidArgument(
                              "TRT fused_token_prune layer building failed."));
  std::vector<std::string> output_tensor_names{"out_slimmed_x", "out_cls_inds"};
  for (size_t i = 0; i < 2; i++) {
    layer->getOutput(i)->setName(output_tensor_names[i].c_str());
    engine_->DeclareOutput(layer, i, output_tensor_names[i]);
  }
  engine_->FreezeNetwork();

#if IS_TRT_VERSION_GE(8600)
  ASSERT_EQ(engine_->engine()->getNbIOTensors(), 6);
#else
  ASSERT_EQ(engine_->engine()->getNbBindings(), 6);
#endif
  LOG(INFO) << "create input";
  std::vector<float> attn_v(16);
  for (int j = 0; j < 4; ++j) {
    for (int k = 0; k < 4; ++k) {
      attn_v[j * 4 + k] = k;
    }
  }
  std::vector<float> x_v(16);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      x_v[i * 4 + j] = 4 - j;
    }
  }
  std::vector<float> mask_v(64);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        mask_v[i * 16 + j * 4 + k] = 1;
      }
    }
  }
  std::vector<float> new_mask_v(16);
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

  auto *attn_gpu_data = inputs_[0].mutable_data<float>(ctx_->GetPlace());
  auto *x_gpu_data = inputs_[1].mutable_data<float>(ctx_->GetPlace());
  auto *mask_gpu_data = inputs_[2].mutable_data<float>(ctx_->GetPlace());
  auto *new_mask_gpu_data = inputs_[3].mutable_data<float>(ctx_->GetPlace());

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

  std::vector<float> slimmed_x_v(8);
  std::vector<int32_t> cls_inds_v;

  LOG(INFO) << "GetOutput";
  GetOutput(slimmed_x_v, cls_inds_v);

  // slimmed_x_v: [[4,3,2,1],[4,3,2,1],[4,3,2,1],[4,3,2,1]] ->
  // [[2,1],[2,1],[2,1],[2,1]]

  ASSERT_EQ(slimmed_x_v[0], 2);
  ASSERT_EQ(slimmed_x_v[1], 1);
  ASSERT_EQ(slimmed_x_v[2], 2);
  ASSERT_EQ(slimmed_x_v[3], 1);
  ASSERT_EQ(slimmed_x_v[4], 2);
  ASSERT_EQ(slimmed_x_v[5], 1);
  ASSERT_EQ(slimmed_x_v[6], 2);
  ASSERT_EQ(slimmed_x_v[7], 1);

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

class TensorRTDynamicTestFusedTokenPruneHalf : public ::testing::Test {
 protected:
  TensorRTDynamicTestFusedTokenPruneHalf()
      : inputs_(), outputs_(), engine_(nullptr), ctx_(nullptr) {}
  void SetUp() override {
    ctx_ = std::make_unique<phi::GPUContext>(phi::GPUPlace(0));
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
    ctx_->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::GPUPinnedPlace())
            .get());
    ctx_->PartialInitWithAllocator();

    std::map<std::string, std::vector<int>> min_input_shape = {
        {"attn", {4, 4}},
        {"x", {4, 4, 1}},
        {"mask", {4, 1, 4, 4}},
        {"new_mask", {4, 1, 2, 2}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"attn", {4, 4}},
        {"x", {4, 4, 1}},
        {"mask", {4, 1, 4, 4}},
        {"new_mask", {4, 1, 2, 2}}};
    std::map<std::string, std::vector<int>> optim_input_shape = {
        {"attn", {4, 4}},
        {"x", {4, 4, 1}},
        {"mask", {4, 1, 4, 4}},
        {"new_mask", {4, 1, 2, 2}}};

    TensorRTEngine::ConstructionParams params;
    params.max_batch_size = 16;
    params.max_workspace_size = 1 << 10;
    params.precision = phi::DataType::FLOAT16;
    params.with_dynamic_shape = true;
    params.min_input_shape = min_input_shape;
    params.max_input_shape = max_input_shape;
    params.optim_input_shape = optim_input_shape;

    engine_ = std::make_unique<TensorRTEngine>(params, NaiveLogger::Global());
    engine_->InitNetwork();
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
      outputs_[i].Resize(common::make_ddim(output_shapes[i]));
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
  std::unique_ptr<TensorRTEngine> engine_;
  std::unique_ptr<phi::GPUContext> ctx_;
};

TEST_F(TensorRTDynamicTestFusedTokenPruneHalf, test_fused_token_prune) {
#if IS_TRT_VERSION_GE(8000)
  auto *attn = engine_->DeclareInput(
      "attn", nvinfer1::DataType::kHALF, nvinfer1::Dims2{-1, 4});
  auto *x = engine_->DeclareInput(
      "x", nvinfer1::DataType::kHALF, nvinfer1::Dims3{-1, 4, 1});
  auto *mask = engine_->DeclareInput(
      "mask", nvinfer1::DataType::kHALF, nvinfer1::Dims4{-1, 1, 4, 4});
  auto *new_mask = engine_->DeclareInput(
      "new_mask", nvinfer1::DataType::kHALF, nvinfer1::Dims4{-1, 1, 2, 2});
  plugin::FusedTokenPrunePluginDynamic *plugin =
      new plugin::FusedTokenPrunePluginDynamic(/*with_fp16*/ true,
                                               /*keep_first_token*/ false,
                                               /*keep_order*/ true,
                                               /*flag_varseqlen*/ false);
  std::vector<nvinfer1::ITensor *> itensors = {attn, x, mask, new_mask};
  auto *layer = engine_->AddDynamicPlugin(itensors.data(), 4, plugin);
  PADDLE_ENFORCE_NOT_NULL(layer,
                          common::errors::InvalidArgument(
                              "TRT fused_token_prune layer building failed."));
  std::vector<std::string> output_tensor_names{"out_slimmed_x", "out_cls_inds"};
  for (size_t i = 0; i < 2; i++) {
    layer->getOutput(i)->setName(output_tensor_names[i].c_str());
    engine_->DeclareOutput(layer, i, output_tensor_names[i]);
  }
  engine_->FreezeNetwork();

#if IS_TRT_VERSION_GE(8600)
  ASSERT_EQ(engine_->engine()->getNbIOTensors(), 6);
#else
  ASSERT_EQ(engine_->engine()->getNbBindings(), 6);
#endif
  LOG(INFO) << "create input";
  std::vector<float16> attn_v(16);
  for (int j = 0; j < 4; ++j) {
    for (int k = 0; k < 4; ++k) {
      attn_v[j * 4 + k] = k;
    }
  }
  std::vector<float16> x_v(16);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      x_v[i * 4 + j] = 4 - j;
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

  std::vector<float> slimmed_x_v(8);
  std::vector<int32_t> cls_inds_v;

  LOG(INFO) << "GetOutput";
  GetOutput(slimmed_x_v, cls_inds_v);

  // slimmed_x_v: [[4,3,2,1],[4,3,2,1],[4,3,2,1],[4,3,2,1]] ->
  // [[2,1],[2,1],[2,1],[2,1]]

  ASSERT_EQ(slimmed_x_v[0], 2);
  ASSERT_EQ(slimmed_x_v[1], 1);
  ASSERT_EQ(slimmed_x_v[2], 2);
  ASSERT_EQ(slimmed_x_v[3], 1);
  ASSERT_EQ(slimmed_x_v[4], 2);
  ASSERT_EQ(slimmed_x_v[5], 1);
  ASSERT_EQ(slimmed_x_v[6], 2);
  ASSERT_EQ(slimmed_x_v[7], 1);

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
#if IS_TRT_VERSION_GE(8000)
class TensorRTDynamicShapeGNTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ctx_ = std::make_unique<phi::GPUContext>(phi::GPUPlace(0));
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
    ctx_->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::GPUPinnedPlace())
            .get());
    ctx_->PartialInitWithAllocator();

    std::map<std::string, std::vector<int>> min_input_shape = {
        {"x", {n_, c_, h_, w_}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"x", {n_, c_, h_, w_}}};
    std::map<std::string, std::vector<int>> optim_input_shape = {
        {"x", {n_, c_, h_, w_}}};
    std::map<std::string, std::vector<int>> min_input_value = {};
    std::map<std::string, std::vector<int>> max_input_value = {};
    std::map<std::string, std::vector<int>> optim_input_value = {};

    TensorRTEngine::ConstructionParams params;
    params.max_batch_size = 16;
    params.max_workspace_size = 1 << 10;
    params.precision = phi::DataType::INT8;
    params.with_dynamic_shape = true;
    params.min_input_shape = min_input_shape;
    params.max_input_shape = max_input_shape;
    params.optim_input_shape = optim_input_shape;

    engine_ = std::make_unique<TensorRTEngine>(params, NaiveLogger::Global());

    engine_->InitNetwork();
  }

  void PrepareInputOutput(const std::vector<float> &input,
                          std::vector<int> output_shape) {
    paddle::framework::TensorFromVector(input, *ctx_, &x_);
    paddle::framework::TensorFromVector(input, *ctx_, &y_);
  }
  void GetOutput(std::vector<float> *output) {
    paddle::framework::TensorToVector(y_, *ctx_, output);
  }

  struct logical_struct {
    int n;
    int c;
    int h;
    int w;
  };

  int nchw(struct logical_struct shape, struct logical_struct index) {
    return index.n * shape.c * shape.h * shape.w + index.c * shape.h * shape.w +
           index.h * shape.w + index.w;
  }

  // this function
  void naive_qdq_cpu(
      float *output, const float *input, int n, float q, float dq) {
    for (int i = 0; i < n; i++) {
      float tmp = input[i];
      int32_t qtmp = std::round(tmp / q);
      qtmp = std::max(-128, qtmp);
      qtmp = std::min(127, qtmp);
      output[i] = qtmp * dq;
    }
  }

  void naive_groupnorm_post_qdq(float *output,
                                const float *input,
                                int n,
                                int c,
                                int h,
                                int w,
                                int groups,
                                float epsilon,
                                float post_scale,
                                const float *scale,
                                const float *bias,
                                bool with_silu) {
    assert(c % groups == 0);
    struct logical_struct shape {
      n, c, h, w
    };

    for (int ni = 0; ni < n; ni++) {
      for (int group_i = 0; group_i < groups; group_i++) {
        int ci_begin = group_i * (c / groups);
        int ci_end = (group_i + 1) * (c / groups);

        float sum = 0.f;
        float q2sum = 0.f;

        for (int ci = ci_begin; ci < ci_end; ci++) {
          for (int hi = 0; hi < h; hi++) {
            for (int wi = 0; wi < w; wi++) {
              struct logical_struct index {
                ni, ci, hi, wi
              };
              float tmp_data = *(input + nchw(shape, index));
              sum += tmp_data;
              q2sum += tmp_data * tmp_data;
            }
          }
        }

        int nums = h * w * c / groups;
        float mean = sum / nums;
        float sigma = sqrtf(q2sum / nums - mean * mean + epsilon);

        for (int ci = ci_begin; ci < ci_end; ci++) {
          for (int hi = 0; hi < h; hi++) {
            for (int wi = 0; wi < w; wi++) {
              struct logical_struct index {
                ni, ci, hi, wi
              };
              float tmp_data = *(input + nchw(shape, index));
              float norm_data = (tmp_data - mean) / sigma;
              *(output + nchw(shape, index)) = norm_data;
            }
          }
        }
      }
    }

    for (int ni = 0; ni < n; ni++) {
      for (int ci = 0; ci < c; ci++) {
        for (int hi = 0; hi < h; hi++) {
          for (int wi = 0; wi < w; wi++) {
            struct logical_struct index {
              ni, ci, hi, wi
            };
            float tmp = *(output + nchw(shape, index));
            float scale_v = scale[ci];
            float bias_v = bias[ci];
            float x = tmp * scale_v + bias_v;
            if (with_silu) {
              x = x / (1 + std::exp(-x));
            }
            *(output + nchw(shape, index)) = x;
          }
        }
      }
    }

    naive_qdq_cpu(output, output, n * c * h * w, post_scale, post_scale);
  }

 protected:
  phi::DenseTensor x_;
  phi::DenseTensor y_;
  std::unique_ptr<TensorRTEngine> engine_;
  std::unique_ptr<phi::GPUContext> ctx_;
  // case from SD
  int n_ = 2;
  int c_ = 320;
  int h_ = 14;
  int w_ = 14;
  int groups_ = 32;
  float epsilon_ = 0.000009999999747378752;
};

// A bug occurred while running int8 mode on v100 :
// [optimizer.cpp::filterQDQFormats::4422] Error Code 2: Internal
// Error (Assertion !n->candidateRequirements.empty() failed. All of the
// candidates were removed, which points to the node being incorrectly marked as
// an int8 node.

/*
TEST_F(TensorRTDynamicShapeGNTest, test_trt_dynamic_shape_groupnorm) {
  float *bias = new float[c_];
  float *scale = new float[c_];
  for (int i = 0; i < c_; i++) {
    bias[i] = (i % 100) / 100.f;
  }
  for (int i = 0; i < c_; i++) {
    scale[i] = (i % 100) / 100.f;
  }

  auto *x = engine_->DeclareInput(
      "x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{n_, c_, h_, w_});

  nvinfer1::Dims scale_dims;
  scale_dims.nbDims = 1;
  scale_dims.d[0] = 1;
  // must set qscale_data = 1.f!
  float qscale_data = 1.f;
  float dqscale_data = 1.f;
  TensorRTEngine::Weight q_weight(nvinfer1::DataType::kFLOAT, &qscale_data,
  1); TensorRTEngine::Weight dq_weight(
      nvinfer1::DataType::kFLOAT, &dqscale_data, 1);

  auto *qscale_tensor =
      TRT_ENGINE_ADD_LAYER(engine_, Constant, scale_dims, q_weight.get())
          ->getOutput(0);
  auto *dqscale_tensor =
      TRT_ENGINE_ADD_LAYER(engine_, Constant, scale_dims, dq_weight.get())
          ->getOutput(0);

  auto *q_layer = TRT_ENGINE_ADD_LAYER(engine_, Quantize, *x,
  *qscale_tensor); q_layer->setAxis(1); auto *q_layer_tensor =
  q_layer->getOutput(0);

  int gn_num = n_ * groups_;
  std::vector<int64_t> mean_shape({gn_num});
  std::vector<int64_t> variance_shape({gn_num});
  bool with_fp16 = true;
  bool with_int8 = true;
  bool with_silu = true;
  plugin::GroupNormPluginDynamic *plugin =
      new plugin::GroupNormPluginDynamic(scale,
                                         c_,
                                         bias,
                                         c_,
                                         epsilon_,
                                         groups_,
                                         mean_shape,
                                         variance_shape,
                                         with_silu,
                                         with_fp16,
                                         with_int8);

  nvinfer1::ILayer *groupnorm_layer =
      engine_->AddDynamicPlugin(&q_layer_tensor, 1, plugin);
  groupnorm_layer->setOutputType(0, nvinfer1::DataType::kINT8);
  auto *gn_tensor = groupnorm_layer->getOutput(0);
  auto *dq_layer =
      TRT_ENGINE_ADD_LAYER(engine_, Dequantize, *gn_tensor, *dqscale_tensor);
  dq_layer->setAxis(1);

  PADDLE_ENFORCE_NOT_NULL(groupnorm_layer,
                          common::errors::InvalidArgument(
                              "TRT GN plugin layer building failed."));

  engine_->DeclareOutput(dq_layer, 0, "y");
  engine_->FreezeNetwork();

  int input_num = n_ * c_ * h_ * w_;
  std::vector<int> shape_v = {n_, c_, h_, w_};

  std::vector<float> x_v(input_num);
  for (int i = 0; i < input_num; i++) {
    x_v[i] = i % 32 - 16;
  }

  PrepareInputOutput(x_v, shape_v);

  engine_->context()->setBindingDimensions(0, nvinfer1::Dims4{n_, c_, h_,
  w_});

  auto *x_gpu_data = x_.data<float>();
  auto *y_gpu_data = y_.mutable_data<float>(ctx_->GetPlace());
  std::vector<void *> buffers(2);
  buffers[0] = reinterpret_cast<void *>(x_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  engine_->Execute(-1, &buffers, ctx_->stream());
  cudaStreamSynchronize(ctx_->stream());

  std::vector<float> y_cpu;
  GetOutput(&y_cpu);
  std::vector<float> y_base(input_num);
  naive_groupnorm_post_qdq(y_base.data(),
                           x_v.data(),
                           n_,
                           c_,
                           h_,
                           w_,
                           groups_,
                           epsilon_,
                           dqscale_data,
                           bias,
                           scale,
                           with_silu);
  float max_diff = -1;
  int right_num = 0;
  for (uint64_t i = 0; i < y_cpu.size(); i++) {
    float diff = std::abs(y_base[i] - y_cpu[i]);
    if (diff < 6e-2) right_num++;
    if (diff > max_diff) max_diff = diff;
  }

  ASSERT_EQ(right_num, input_num);

  delete[] bias;
  delete[] scale;
  return;
}
*/
#endif
}  // namespace paddle::inference::tensorrt
