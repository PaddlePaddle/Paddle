/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_bn_activation_op.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using LoDTensor = phi::DenseTensor;

void FusedBatchNormActOp::InferShape(framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE_EQ(ctx->HasInput("X"),
                    true,
                    platform::errors::InvalidArgument(
                        "Input(X) of BatchNormOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("Scale"),
                    true,
                    platform::errors::InvalidArgument(
                        "Input(Scale) of BatchNormOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("Bias"),
                    true,
                    platform::errors::InvalidArgument(
                        "Input(Bias) of BatchNormOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("Mean"),
                    true,
                    platform::errors::InvalidArgument(
                        "Input(Mean) of BatchNormOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("Variance"),
                    true,
                    platform::errors::InvalidArgument(
                        "Input(Variance) of BatchNormOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasOutput("Y"),
                    true,
                    platform::errors::InvalidArgument(
                        "Output(Y) of BatchNormOp should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasOutput("MeanOut"),
                    true,
                    platform::errors::InvalidArgument(
                        "Output(MeanOut) of BatchNormOp should not be null."));
  PADDLE_ENFORCE_EQ(
      ctx->HasOutput("VarianceOut"),
      true,
      platform::errors::InvalidArgument(
          "Output(VarianceOut) of BatchNormOp should not be null."));
  PADDLE_ENFORCE_EQ(
      ctx->HasOutput("SavedMean"),
      true,
      platform::errors::InvalidArgument(
          "Output(SavedMean) of BatchNormOp should not be null."));
  PADDLE_ENFORCE_EQ(
      ctx->HasOutput("SavedVariance"),
      true,
      platform::errors::InvalidArgument(
          "Output(SavedVariance) of BatchNormOp should not be null."));

  // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
  PADDLE_ENFORCE_EQ(ctx->Inputs("Mean")[0],
                    ctx->Outputs("MeanOut")[0],
                    platform::errors::PreconditionNotMet(
                        "Mean and MeanOut should share the same memory"));
  PADDLE_ENFORCE_EQ(
      ctx->Inputs("Variance")[0],
      ctx->Outputs("VarianceOut")[0],
      platform::errors::PreconditionNotMet(
          "Variance and VarianceOut should share the same memory"));

  const auto x_dims = ctx->GetInputDim("X");

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      platform::errors::PreconditionNotMet("ShapeError: the dimension of input "
                                           "X must greater than or equal to 2."
                                           "But received: the shape of input X "
                                           "= [%s], the dimension of input X ="
                                           "[%d]",
                                           x_dims,
                                           x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(),
      5,
      platform::errors::PreconditionNotMet("ShapeError: the dimension of input "
                                           "X must smaller than or equal to 5."
                                           "But received: the shape of input X "
                                           "= [%s], the dimension of input X ="
                                           "[%d]",
                                           x_dims,
                                           x_dims.size()));

  const int64_t C = x_dims[x_dims.size() - 1];

  auto scale_dim = ctx->GetInputDim("Scale");
  auto bias_dim = ctx->GetInputDim("Bias");

  PADDLE_ENFORCE_EQ(
      scale_dim.size(),
      1UL,
      platform::errors::PreconditionNotMet(
          "ShapeError: the dimension of scale must equal to 1."
          "But received: the shape of scale is [%s], the dimension "
          "of scale is [%d]",
          scale_dim,
          scale_dim.size()));
  PADDLE_ENFORCE_EQ(bias_dim.size(),
                    1UL,
                    platform::errors::PreconditionNotMet(
                        "ShapeError: the dimension of bias must equal to 1."
                        "But received: the shape of bias is [%s],the dimension "
                        "of bias is [%d]",
                        bias_dim,
                        bias_dim.size()));

  bool check = true;
  if ((!ctx->IsRuntime()) &&
      (phi::product(scale_dim) <= 0 || phi::product(bias_dim) <= 0)) {
    check = false;
  }

  if (check) {
    PADDLE_ENFORCE_EQ(scale_dim[0],
                      C,
                      platform::errors::PreconditionNotMet(
                          "ShapeError: the shape of scale must equal to [%d]"
                          "But received: the shape of scale is [%d]",
                          C,
                          scale_dim[0]));
    PADDLE_ENFORCE_EQ(bias_dim[0],
                      C,
                      platform::errors::PreconditionNotMet(
                          "ShapeError: the shape of bias must equal to [%d]"
                          "But received: the shape of bias is [%d]",
                          C,
                          bias_dim[0]));
  }
  ctx->SetOutputDim("Y", x_dims);
  ctx->SetOutputDim("MeanOut", {C});
  ctx->SetOutputDim("VarianceOut", {C});
  ctx->SetOutputDim("SavedMean", {C});
  ctx->SetOutputDim("SavedVariance", {C});
  ctx->ShareLoD("X", "Y");
}

framework::OpKernelType FusedBatchNormActOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
  // By default, the type of the scale, bias, mean,
  // and var tensors should both be float. (For float or float16 input tensor)
  // or double (For double input tensor).
  auto bn_param_type = framework::proto::VarType::FP32;
  if (input_data_type == framework::proto::VarType::FP64) {
    bn_param_type = framework::proto::VarType::FP64;
  }
  PADDLE_ENFORCE_EQ(bn_param_type,
                    framework::TransToProtoVarType(
                        ctx.Input<phi::DenseTensor>("Scale")->dtype()),
                    platform::errors::PreconditionNotMet(
                        "Scale input should be of float type"));
  PADDLE_ENFORCE_EQ(bn_param_type,
                    framework::TransToProtoVarType(
                        ctx.Input<phi::DenseTensor>("Bias")->dtype()),
                    platform::errors::PreconditionNotMet(
                        "Bias input should be of float type"));
  PADDLE_ENFORCE_EQ(bn_param_type,
                    framework::TransToProtoVarType(
                        ctx.Input<phi::DenseTensor>("Mean")->dtype()),
                    platform::errors::PreconditionNotMet(
                        "Mean input should be of float type"));
  PADDLE_ENFORCE_EQ(bn_param_type,
                    framework::TransToProtoVarType(
                        ctx.Input<phi::DenseTensor>("Variance")->dtype()),
                    platform::errors::PreconditionNotMet(
                        "Variance input should be of float type"));

  framework::LibraryType library = framework::LibraryType::kPlain;
  phi::DataLayout layout = phi::DataLayout::kAnyLayout;

  return framework::OpKernelType(
      input_data_type, ctx.GetPlace(), layout, library);
}

void FusedBatchNormActOpMaker::Make() {
  AddAttr<float>("momentum", "").SetDefault(0.9);
  AddAttr<float>("epsilon", "")
      .SetDefault(1e-5)
      .AddCustomChecker([](const float &epsilon) {
        PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f,
                          true,
                          platform::errors::InvalidArgument(
                              "Attr(epsilon) should be between 0.0 and 0.001, "
                              "but received value is %f.",
                              epsilon));
      });
  AddAttr<std::string>("act_type", "The activation type to be fused.")
      .SetDefault("relu");
  AddInput("X", "The input tensor");
  AddInput("Scale",
           "Scale is a 1-dimensional tensor of size C "
           "that is applied to the output");
  AddInput("Bias",
           "Bias is a 1-dimensional tensor of size C "
           "that is applied to the output");
  AddInput("Mean",
           "The global mean (for training) or "
           "estimated mean (for testing)");
  AddInput("Variance",
           "The global variance (for training) "
           "or estimated Variance (for testing)");
  AddOutput("Y", "result after normalization");
  AddOutput("MeanOut",
            "Share memory with Mean. "
            "Store the global mean when training");
  AddOutput("VarianceOut",
            "Share memory with Variance. "
            "Store the global Variance when training");
  AddOutput("SavedMean",
            "Mean of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate();
  AddOutput("SavedVariance",
            "Variance of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate();
  AddOutput("ReserveSpace",
            "Reserve GPU space for triggering the new semi-persistent "
            "NHWC kernel");
  AddComment(R"DOC(
Fused Batch Normalization with activation.

Batch Norm has been implemented as discussed in the paper:
https://arxiv.org/pdf/1502.03167.pdf
Batch Norm can be used as a normalizer function for conv2d and fully_connected operations.
Now, the required data format for FusedBatchNormActOp is NHWC `[batch, in_height, in_width, in_channels]`.

)DOC");
}

void FusedBatchNormActGradOp::InferShape(
    framework::InferShapeContext *ctx) const {
  // check input
  PADDLE_ENFORCE_EQ(
      ctx->HasInput("X"),
      true,
      platform::errors::InvalidArgument("Input(X) should not be null."));
  PADDLE_ENFORCE_EQ(
      ctx->HasInput("Scale"),
      true,
      platform::errors::InvalidArgument("Input(Scale) should not be null."));
  PADDLE_ENFORCE_EQ(
      ctx->HasInput(framework::GradVarName("Y")),
      true,
      platform::errors::InvalidArgument("Input(Y@GRAD) should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("SavedMean"),
                    true,
                    platform::errors::InvalidArgument(
                        "Input(SavedMean) should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasInput("SavedVariance"),
                    true,
                    platform::errors::InvalidArgument(
                        "Input(SavedVariance) should not be null"));

  // check output
  PADDLE_ENFORCE_EQ(
      ctx->HasOutput(framework::GradVarName("X")),
      true,
      platform::errors::InvalidArgument("Output(X@GRAD) should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Scale")),
                    true,
                    platform::errors::InvalidArgument(
                        "Output(Scale@GRAD) should not be null."));
  PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Bias")),
                    true,
                    platform::errors::InvalidArgument(
                        "Output(Bias@GRAD) should not be null."));

  const auto x_dims = ctx->GetInputDim("X");
  const int C = x_dims[x_dims.size() - 1];

  ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  // has_scale_grad == has_bias_grad, judge has_scale_grad is enough
  ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
  ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
}

framework::OpKernelType FusedBatchNormActGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar(framework::GradVarName("Y"));
  if (var == nullptr) {
    PADDLE_THROW(platform::errors::NotFound(
        "Can not find Y@GRAD in the execution context."));
  }
  const Tensor *t = nullptr;
  if (var->IsType<Tensor>()) {
    t = &var->Get<Tensor>();
  } else if (var->IsType<LoDTensor>()) {
    t = &var->Get<LoDTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW(
        platform::errors::NotFound("Can not get the tensor value of Y@GRAD."));
  }

  framework::LibraryType library = framework::LibraryType::kPlain;
  phi::DataLayout layout = phi::DataLayout::kAnyLayout;

  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"),
      ctx.GetPlace(),
      layout,
      library);
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_batch_norm_act,
    ops::FusedBatchNormActOp,
    ops::FusedBatchNormActOpMaker,
    ops::FusedBatchNormActOpInferVarType,
    ops::FusedBatchNormActGradOpMaker<paddle::framework::OpDesc>,
    ops::FusedBatchNormActGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_batch_norm_act_grad, ops::FusedBatchNormActGradOp);
