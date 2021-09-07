/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/resnet_unit_op.h"

namespace paddle {
namespace operators {

void ResNetUnitOp::InferShape(framework::InferShapeContext *ctx) const {
  // check input
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ResNetUnitOp");
  OP_INOUT_CHECK(ctx->HasInput("FilterX"), "Input", "FilterX", "ResNetUnitOp");
  OP_INOUT_CHECK(ctx->HasInput("ScaleX"), "Input", "ScaleX", "ResNetUnitOp");
  OP_INOUT_CHECK(ctx->HasInput("BiasX"), "Input", "BiasX", "ResNetUnitOp");
  if (ctx->Attrs().Get<bool>("fused_add")) {
    OP_INOUT_CHECK(ctx->HasInput("Z"), "Input", "Z", "ResNetUnitOp");
  }
  if (ctx->Attrs().Get<bool>("has_shortcut")) {
    OP_INOUT_CHECK(ctx->HasInput("FilterZ"), "Input", "FilterZ",
                   "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasInput("ScaleZ"), "Input", "ScaleZ", "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasInput("BiasZ"), "Input", "BiasZ", "ResNetUnitOp");
  }

  // check output
  OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "ResNetUnitOp");

  // TODO(zhangzheng): check dims for input and output
  const auto x_dims = ctx->GetInputDim("X");
  // TODO(zhangzheng): infer shape of output
  const auto w_dims = ctx->GetInputDim("FilterX");
  int batch = x_dims[0];
  int output_channel = w_dims[0];
  int input_channel = w_dims[1];
  int filter_size = w_dims[2];
  int stride = ctx->Attrs().Get<int>("stride");
  int pad = ctx->Attrs().Get<int>("pad");
  int out_h = (x_dims[2] + pad * 2 - filter_size) / stride + 1;
  int out_w = (x_dims[3] + pad * 2 - filter_size) / stride + 1;
  std::vector<int> out_shape = {batch, output_channel, out_h, out_w};
  // shape of bitmask
  int C = input_channel;
  int64_t NHW = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                std::multiplies<int>()) /
                output_channel;
  int32_t C_int32Elems = ((C + 63) & ~63) / 32;
  int32_t NHW_int32Elems = (NHW + 31) & ~31;
  std::vector<int> bitmask_shape = {NHW_int32Elems, C_int32Elems, 1};

  auto y_dims = framework::make_ddim(out_shape);
  auto bitmask_dims = framework::make_ddim(bitmask_shape);
  auto bn_param_dims = framework::make_ddim({1, output_channel, 1, 1});
  ctx->SetOutputDim("Y", y_dims);
  ctx->SetOutputDim("BitMask", bitmask_dims);
  ctx->SetOutputDim("ConvX", y_dims);
  ctx->SetOutputDim("SumX", bn_param_dims);
  ctx->SetOutputDim("SqSumX", bn_param_dims);
  ctx->SetOutputDim("SavedMeanX", bn_param_dims);
  ctx->SetOutputDim("SavedInvstdX", bn_param_dims);
  ctx->SetOutputDim("RunningMeanX", bn_param_dims);
  ctx->SetOutputDim("RunningVarX", bn_param_dims);
  ctx->SetOutputDim("EqScaleX", bn_param_dims);
  ctx->SetOutputDim("EqBiasX", bn_param_dims);
  if (ctx->Attrs().Get<bool>("has_shortcut")) {
    ctx->SetOutputDim("ConvZ", y_dims);
    ctx->SetOutputDim("SumZ", bn_param_dims);
    ctx->SetOutputDim("SqSumZ", bn_param_dims);
    ctx->SetOutputDim("SavedMeanZ", bn_param_dims);
    ctx->SetOutputDim("SavedInvstdZ", bn_param_dims);
    ctx->SetOutputDim("RunningMeanZ", bn_param_dims);
    ctx->SetOutputDim("RunningVarZ", bn_param_dims);
    ctx->SetOutputDim("EqScaleZ", bn_param_dims);
    ctx->SetOutputDim("EqBiasZ", bn_param_dims);
  }
}

framework::OpKernelType ResNetUnitOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
  return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                 library);
}

void ResNetUnitOpMaker::Make() {
  AddInput("X", "The input 1 tensor");
  AddInput("FilterX", "The filter tensor of input 1");
  AddInput("ScaleX", "The bn scale tensor of input 1");
  AddInput("BiasX", "The bn bias tensor of input 1");
  AddInput("Z", "The input 2 tensor");
  AddInput("FilterZ", "The filter tensor of input 2");
  AddInput("ScaleZ", "The bn scale tensor of input 2");
  AddInput("BiasZ", "The bn bias tensor of input 2");
  AddOutput("Y", "The result of the resnet unit");
  AddOutput("BitMask", "The bitmask");
  AddOutput("ConvX", "The output of x after conv");
  AddOutput("SumX", "The sum of conv_x");
  AddOutput("SqSumX", "The square of sum of conv_x");
  AddOutput("SavedMeanX", "The output of saved mean of x");
  AddOutput("SavedInvstdX", "The output of saved invstd of x");
  AddOutput("RunningMeanX", "The output of running mean of x");
  AddOutput("RunningVarX", "The output of running var of x");
  AddOutput("EqScaleX", "The output of equiv scale of x");
  AddOutput("EqBiasX", "The output of equiv bias of x");
  AddOutput("ConvZ", "The output of z after conv");
  AddOutput("SumZ", "The sum of conv_z");
  AddOutput("SqSumZ", "The square of sum of conv_z");
  AddOutput("SavedMeanZ", "The output of saved mean of z");
  AddOutput("SavedInvstdZ", "The output of saved invstd of z");
  AddOutput("RunningMeanZ", "The output of running mean of z");
  AddOutput("RunningVarZ", "The output of running var of z");
  AddOutput("EqScaleZ", "The output of equiv scale of z");
  AddOutput("EqBiasZ", "The output of equiv bias of z");
  AddAttr<int>("ele_count", "");
  AddAttr<int>("stride", "").SetDefault(1);
  AddAttr<int>("pad", "").SetDefault(0);
  AddAttr<int>("dilate", "").SetDefault(1);
  AddAttr<int>("group", "").SetDefault(1);
  AddAttr<float>("momentum", "").SetDefault(0.9);
  AddAttr<float>("epsilon", "").SetDefault(1e-5);
  AddAttr<std::string>("conv_format", "").SetDefault("NHWC");
  AddAttr<std::string>("bn_format", "").SetDefault("NHWC");
  AddAttr<bool>("fused_add", "").SetDefault(false);
  AddAttr<bool>("has_shortcut", "").SetDefault(false);
  AddAttr<std::string>("act_type", "The activation type to be fused.")
      .SetDefault("relu");
  AddComment(R"DOC(
****TODO****.
)DOC");
}

void ResNetUnitGradOp::InferShape(framework::InferShapeContext *ctx) const {
  // check input
  // check output
}

framework::OpKernelType ResNetUnitGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar(framework::GradVarName("Y"));
  if (var == nullptr) {
    PADDLE_THROW(platform::errors::NotFound(
        "Can not find Y@GRAD in the execution context."));
  }

  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;

  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(), layout,
      library);
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(resnet_unit, ops::ResNetUnitOp, ops::ResNetUnitOpMaker,
                  ops::ResNetUnitOpInferVarType,
                  ops::ResNetUnitGradOpMaker<paddle::framework::OpDesc>,
                  ops::ResNetUnitGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(resnet_unit_grad, ops::ResNetUnitGradOp);
