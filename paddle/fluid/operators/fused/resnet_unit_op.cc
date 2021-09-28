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
  OP_INOUT_CHECK(ctx->HasInput("MeanX"), "Input", "MeanX", "ResNetUnitOp");
  OP_INOUT_CHECK(ctx->HasInput("VarX"), "Input", "VarX", "ResNetUnitOp");
  if (ctx->Attrs().Get<bool>("fused_add")) {
    OP_INOUT_CHECK(ctx->HasInput("Z"), "Input", "Z", "ResNetUnitOp");
  }
  if (ctx->Attrs().Get<bool>("has_shortcut")) {
    OP_INOUT_CHECK(ctx->HasInput("FilterZ"), "Input", "FilterZ",
                   "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasInput("ScaleZ"), "Input", "ScaleZ", "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasInput("BiasZ"), "Input", "BiasZ", "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasInput("MeanZ"), "Input", "MeanZ", "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasInput("VarZ"), "Input", "VarZ", "ResNetUnitOp");
  }

  // check output
  OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "ResNetUnitOp");
  // OP_INOUT_CHECK(ctx->HasOutput("BitMask"), "Output", "BitMask",
  //                "ResNetUnitOp");
  OP_INOUT_CHECK(ctx->HasOutput("ConvX"), "Output", "ConvX", "ResNetUnitOp");
  // OP_INOUT_CHECK(ctx->HasOutput("SumX"), "Output", "SumX", "ResNetUnitOp");
  // OP_INOUT_CHECK(ctx->HasOutput("SqSumX"), "Output", "SqSumX",
  // "ResNetUnitOp");
  OP_INOUT_CHECK(ctx->HasOutput("SavedMeanX"), "Output", "SavedMeanX",
                 "ResNetUnitOp");
  OP_INOUT_CHECK(ctx->HasOutput("SavedInvstdX"), "Output", "SavedInvstdX",
                 "ResNetUnitOp");
  OP_INOUT_CHECK(ctx->HasOutput("RunningMeanX"), "Output", "RunningMeanX",
                 "ResNetUnitOp");
  OP_INOUT_CHECK(ctx->HasOutput("RunningVarX"), "Output", "RunningVarX",
                 "ResNetUnitOp");
  // OP_INOUT_CHECK(ctx->HasOutput("EqScaleX"), "Output", "EqScaleX",
  //                "ResNetUnitOp");
  // OP_INOUT_CHECK(ctx->HasOutput("EqBiasX"), "Output", "EqBiasX",
  //                "ResNetUnitOp");
  if (ctx->Attrs().Get<bool>("has_shortcut")) {
    OP_INOUT_CHECK(ctx->HasOutput("ConvZ"), "Output", "ConvZ", "ResNetUnitOp");
    // OP_INOUT_CHECK(ctx->HasOutput("SumZ"), "Output", "SumZ", "ResNetUnitOp");
    // OP_INOUT_CHECK(ctx->HasOutput("SqSumZ"), "Output", "SqSumZ",
    //                "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasOutput("SavedMeanZ"), "Output", "SavedMeanZ",
                   "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasOutput("SavedInvstdZ"), "Output", "SavedInvstdZ",
                   "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasOutput("RunningMeanZ"), "Output", "RunningMeanZ",
                   "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasOutput("RunningVarZ"), "Output", "RunningVarZ",
                   "ResNetUnitOp");
    // OP_INOUT_CHECK(ctx->HasOutput("EqScaleZ"), "Output", "EqScaleZ",
    //                "ResNetUnitOp");
    // OP_INOUT_CHECK(ctx->HasOutput("EqBiasZ"), "Output", "EqBiasZ",
    //                "ResNetUnitOp");
  }

  // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
  PADDLE_ENFORCE_EQ(ctx->Inputs("MeanX")[0], ctx->Outputs("RunningMeanX")[0],
                    platform::errors::InvalidArgument(
                        "Mean and MeanOut should share the same memory"));
  PADDLE_ENFORCE_EQ(
      ctx->Inputs("VarX")[0], ctx->Outputs("RunningVarX")[0],
      platform::errors::InvalidArgument(
          "Variance and VarianceOut should share the same memory"));
  if (ctx->Attrs().Get<bool>("has_shortcut")) {
    PADDLE_ENFORCE_EQ(ctx->Inputs("MeanZ")[0], ctx->Outputs("RunningMeanZ")[0],
                      platform::errors::InvalidArgument(
                          "Mean and MeanOut should share the same memory"));
    PADDLE_ENFORCE_EQ(
        ctx->Inputs("VarZ")[0], ctx->Outputs("RunningVarZ")[0],
        platform::errors::InvalidArgument(
            "Variance and VarianceOut should share the same memory"));
  }

  // TODO(zhangzheng): check dims for input and output
  const auto x_dims = ctx->GetInputDim("X");
  const auto w_dims = ctx->GetInputDim("FilterX");
  int batch = x_dims[0];
  int output_channel = w_dims[0];
  int filter_size = w_dims[2];
  int stride = ctx->Attrs().Get<int>("stride");
  int pad = ctx->Attrs().Get<int>("pad");
  int out_h = (x_dims[1] + pad * 2 - filter_size) / stride + 1;
  int out_w = (x_dims[2] + pad * 2 - filter_size) / stride + 1;
  std::vector<int> out_shape = {batch, out_h, out_w, output_channel};
  // shape of bitmask
  // int C = output_channel;
  // int64_t NHW = std::accumulate(out_shape.begin(), out_shape.end(), 1,
  //                               std::multiplies<int>()) /
  //               output_channel;
  // int32_t C_int32Elems = ((C + 63) & ~63) / 32;
  // int32_t NHW_int32Elems = (NHW + 31) & ~31;
  // std::vector<int> bitmask_shape = {NHW_int32Elems, C_int32Elems, 1};
  // printf("==============%d, %d\n", NHW_int32Elems, C_int32Elems);

  auto y_dims = framework::make_ddim(out_shape);
  // auto bitmask_dims = framework::make_ddim(bitmask_shape);
  auto bn_param_dims = framework::make_ddim({1, 1, 1, output_channel});
  ctx->SetOutputDim("Y", y_dims);
  // ctx->SetOutputDim("BitMask", bitmask_dims);
  ctx->SetOutputDim("ConvX", y_dims);
  // ctx->SetOutputDim("SumX", bn_param_dims);
  // ctx->SetOutputDim("SqSumX", bn_param_dims);
  ctx->SetOutputDim("SavedMeanX", bn_param_dims);
  ctx->SetOutputDim("SavedInvstdX", bn_param_dims);
  ctx->SetOutputDim("RunningMeanX", bn_param_dims);
  ctx->SetOutputDim("RunningVarX", bn_param_dims);
  // ctx->SetOutputDim("EqScaleX", bn_param_dims);
  // ctx->SetOutputDim("EqBiasX", bn_param_dims);
  if (ctx->Attrs().Get<bool>("has_shortcut")) {
    ctx->SetOutputDim("ConvZ", y_dims);
    // ctx->SetOutputDim("SumZ", bn_param_dims);
    // ctx->SetOutputDim("SqSumZ", bn_param_dims);
    ctx->SetOutputDim("SavedMeanZ", bn_param_dims);
    ctx->SetOutputDim("SavedInvstdZ", bn_param_dims);
    ctx->SetOutputDim("RunningMeanZ", bn_param_dims);
    ctx->SetOutputDim("RunningVarZ", bn_param_dims);
    // ctx->SetOutputDim("EqScaleZ", bn_param_dims);
    // ctx->SetOutputDim("EqBiasZ", bn_param_dims);
  }
}

framework::OpKernelType ResNetUnitOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
  // By default, the type of the scale, bias, mean,
  // and var tensors should be float when input tensor's dtype is float16.
  auto bn_param_type = framework::proto::VarType::FP32;

  PADDLE_ENFORCE_EQ(
      bn_param_type, ctx.Input<Tensor>("ScaleX")->type(),
      platform::errors::InvalidArgument("Scale input should be of float type"));
  PADDLE_ENFORCE_EQ(
      bn_param_type, ctx.Input<Tensor>("BiasX")->type(),
      platform::errors::InvalidArgument("Bias input should be of float type"));
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
  AddInput("MeanX", "The bn mean tensor of input 1");
  AddInput("VarX", "The bn var tensor of input 1");
  AddInput("Z", "The input 2 tensor").AsDispensable();
  AddInput("FilterZ", "The filter tensor of input 2").AsDispensable();
  AddInput("ScaleZ", "The bn scale tensor of input 2").AsDispensable();
  AddInput("BiasZ", "The bn bias tensor of input 2").AsDispensable();
  AddInput("MeanZ", "The bn mean tensor of input 2").AsDispensable();
  AddInput("VarZ", "The bn var tensor of input 2").AsDispensable();
  AddOutput("Y", "The result of the resnet unit");
  // AddOutput("BitMask", "The bitmask");
  AddOutput("ConvX", "The output of x after conv");
  // AddOutput("SumX", "The sum of conv_x");
  // AddOutput("SqSumX", "The square of sum of conv_x");
  AddOutput("SavedMeanX", "The output of saved mean of x");
  AddOutput("SavedInvstdX", "The output of saved invstd of x");
  AddOutput("RunningMeanX", "The output of running mean of x");
  AddOutput("RunningVarX", "The output of running var of x");
  // AddOutput("EqScaleX", "The output of equiv scale of x");
  // AddOutput("EqBiasX", "The output of equiv bias of x");
  AddOutput("ConvZ", "The output of z after conv").AsDispensable();
  // AddOutput("SumZ", "The sum of conv_z").AsDispensable();
  // AddOutput("SqSumZ", "The square of sum of conv_z").AsDispensable();
  AddOutput("SavedMeanZ", "The output of saved mean of z").AsDispensable();
  AddOutput("SavedInvstdZ", "The output of saved invstd of z").AsDispensable();
  AddOutput("RunningMeanZ", "The output of running mean of z").AsDispensable();
  AddOutput("RunningVarZ", "The output of running var of z").AsDispensable();
  // AddOutput("EqScaleZ", "The output of equiv scale of z").AsDispensable();
  // AddOutput("EqBiasZ", "The output of equiv bias of z").AsDispensable();
  AddAttr<int>("stride", "").SetDefault(1);
  AddAttr<int>("stride_z", "").SetDefault(1);
  AddAttr<int>("pad", "").SetDefault(0);
  AddAttr<int>("dilate", "").SetDefault(1);
  AddAttr<int>("group", "").SetDefault(1);
  AddAttr<float>("momentum", "").SetDefault(0.9);
  AddAttr<float>("epsilon", "").SetDefault(1e-5);
  AddAttr<std::string>("conv_format", "").SetDefault("NHWC");
  AddAttr<std::string>("bn_format", "").SetDefault("NHWC");
  AddAttr<bool>("fused_add", "").SetDefault(false);
  AddAttr<bool>("has_shortcut", "").SetDefault(false);
  AddAttr<bool>("use_global_stats", "").SetDefault(false);
  AddAttr<bool>("use_addto", "").SetDefault(false);
  AddAttr<std::string>("act_type", "The activation type to be fused.")
      .SetDefault("relu");
  AddComment(R"DOC(
****TODO****.
)DOC");
}

void ResNetUnitGradOp::InferShape(framework::InferShapeContext *ctx) const {
  // check input
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ResNetUnitGradOp");
  OP_INOUT_CHECK(ctx->HasInput("FilterX"), "Input", "FilterX",
                 "ResNetUnitGradOp");
  OP_INOUT_CHECK(ctx->HasInput("ConvX"), "Input", "ConvX", "ResNetUnitGradOp");
  OP_INOUT_CHECK(ctx->HasInput("ScaleX"), "Input", "ScaleX",
                 "ResNetUnitGradOp");
  OP_INOUT_CHECK(ctx->HasInput("BiasX"), "Input", "BiasX", "ResNetUnitGradOp");
  OP_INOUT_CHECK(ctx->HasInput("SavedMeanX"), "Input", "SavedMeanX",
                 "ResNetUnitGradOp");
  OP_INOUT_CHECK(ctx->HasInput("SavedInvstdX"), "Input", "SavedInvstdX",
                 "ResNetUnitGradOp");
  if (ctx->Attrs().Get<bool>("fused_add")) {
    OP_INOUT_CHECK(ctx->HasInput("Z"), "Input", "Z", "ResNetUnitGradOp");
  }
  if (ctx->Attrs().Get<bool>("has_shortcut")) {
    OP_INOUT_CHECK(ctx->HasInput("FilterZ"), "Input", "FilterZ",
                   "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("ConvZ"), "Input", "ConvZ",
                   "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("ScaleZ"), "Input", "ScaleZ",
                   "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("BiasZ"), "Input", "BiasZ",
                   "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("SavedMeanZ"), "Input", "SavedMeanZ",
                   "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("SavedInvstdZ"), "Input", "SavedInvstdZ",
                   "ResNetUnitGradOp");
  }
  OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ResNetUnitGradOp");
  // OP_INOUT_CHECK(ctx->HasInput("BitMask"), "Input", "BitMask",
  //                "ResNetUnitGradOp");
  OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")), "Input",
                 framework::GradVarName("Y"), "ResNetUnitGradOp");

  // check output
  OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                 framework::GradVarName("X"), "ResNetUnitGradOp");
  OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("FilterX")), "Output",
                 framework::GradVarName("FilterX"), "ResNetUnitGradOp");
  OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("ScaleX")), "Output",
                 framework::GradVarName("ScaleX"), "ResNetUnitGradOp");
  OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BiasX")), "Output",
                 framework::GradVarName("BiasX"), "ResNetUnitGradOp");
  // OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("ConvX")), "Output",
  //                framework::GradVarName("ConvX"), "ResNetUnitGradOp");
  if (ctx->Attrs().Get<bool>("fused_add")) {
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Z")), "Output",
                   framework::GradVarName("Z"), "ResNetUnitGradOp");
  }
  if (ctx->Attrs().Get<bool>("has_shortcut")) {
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("FilterZ")), "Output",
                   framework::GradVarName("FilterZ"), "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("ScaleZ")), "Output",
                   framework::GradVarName("ScaleZ"), "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BiasZ")), "Output",
                   framework::GradVarName("BiasZ"), "ResNetUnitGradOp");
    // OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("ConvZ")), "Output",
    //                framework::GradVarName("ConvZ"), "ResNetUnitGradOp");
  }
  const auto x_dims = ctx->GetInputDim("X");
  const auto filter_x_dims = ctx->GetInputDim("FilterX");
  // const auto y_dims = ctx->GetInputDim("Y");
  const auto param_dims = ctx->GetInputDim("ScaleX");
  ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  ctx->SetOutputDim(framework::GradVarName("FilterX"), filter_x_dims);
  ctx->SetOutputDim(framework::GradVarName("ScaleX"), param_dims);
  ctx->SetOutputDim(framework::GradVarName("BiasX"), param_dims);
  // ctx->SetOutputDim(framework::GradVarName("ConvX"), y_dims);
  if (ctx->Attrs().Get<bool>("fused_add")) {
    const auto z_dims = ctx->GetInputDim("Z");
    ctx->SetOutputDim(framework::GradVarName("Z"), z_dims);
  }
  if (ctx->Attrs().Get<bool>("has_shortcut")) {
    const auto filter_z_dims = ctx->GetInputDim("FilterZ");
    ctx->SetOutputDim(framework::GradVarName("FilterZ"), filter_z_dims);
    ctx->SetOutputDim(framework::GradVarName("ScaleZ"), param_dims);
    ctx->SetOutputDim(framework::GradVarName("BiasZ"), param_dims);
    // ctx->SetOutputDim(framework::GradVarName("ConvZ"), y_dims);
  }
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
