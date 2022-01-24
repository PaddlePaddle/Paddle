/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/operators/unfold_op.h"

namespace paddle {
namespace operators {

class UnfoldOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Tensor, "
             "the input of unfold op. "
             "The format of X is [N, C_in, H, W], "
             "where N is the batch size, C_in is the input channels, "
             "H is the height and W is the width");
    AddOutput(
        "Y",
        "Tensor, "
        "the output of unfold op. "
        "The format of Y is [N, C_in*filter_height*filter_width, "
        "output_height*output_width], where N is the batch size, "
        "C_in is the input channels of X, filter_height and filter_width is "
        "height and width of the filtering kernel, output_height and "
        "output_width "
        "is the calculated height and width of output feature map.");
    AddAttr<std::vector<int>>(
        "kernel_sizes",
        "vector<int>, the kernel sizes of the convolution operator.");
    AddAttr<std::vector<int>>(
        "strides", "vector<int>, the strides of the convolution operator.");
    AddAttr<std::vector<int>>(
        "paddings",
        "vector<int>, the paddings applied to pad the feature map.");
    AddAttr<std::vector<int>>(
        "dilations", "vector<int>, the dilations of the convolution operator.");
    AddComment(R"DOC(
**Unfold Operator**

This Operator is used to extract sliding local blocks from a batched input tensor, also known
as im2col when operated on batched 2D image tensor. For each block under the convolution filter,
all element will be rearranged as a column. While the convolution filter sliding over the input
feature map, a series of such columns will be formed. 
    )DOC");
  }
};

class UnfoldOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of UnfoldOp should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Y"), true,
        platform::errors::NotFound("Output(Y) of UnfoldOp should not be null"));
    auto in_dims = ctx->GetInputDim("X");
    std::vector<int> kernel_sizes =
        ctx->Attrs().Get<std::vector<int>>("kernel_sizes");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    std::vector<int> dilations =
        ctx->Attrs().Get<std::vector<int>>("dilations");

    // Only [N, C, H, W] input supported now
    PADDLE_ENFORCE_EQ(
        in_dims.size(), 4,
        platform::errors::InvalidArgument(
            "Input should be 4-D tensor of format [N, C, H, W], but get %u",
            in_dims.size()));
    PADDLE_ENFORCE_EQ(
        in_dims.size() - kernel_sizes.size(), 2U,
        platform::errors::InvalidArgument(
            "The dims of X should be larger than that of kernel_sizes "
            "by a number of 2, due to the batch size and input channel dim. "
            "But recieved dims(X:%u) - dims(kernel_sizes:%u) != 2",
            in_dims.size(), kernel_sizes.size()));
    PADDLE_ENFORCE_EQ(
        strides.size(), kernel_sizes.size(),
        platform::errors::InvalidArgument(
            "The dims of strides should be the same with that of kernel_sizes. "
            "But recieved dims(strides: %u) != dims(kernel_sizes: %u).",
            strides.size(), kernel_sizes.size()));
    PADDLE_ENFORCE_EQ(
        paddings.size(), 2 * strides.size(),
        platform::errors::InvalidArgument(
            "The dims of paddings should be 2 times of that of strides. "
            "But recieved dims(paddings: %u) != 2*dims(strides: %u).",
            paddings.size(), strides.size()));
    PADDLE_ENFORCE_EQ(
        strides.size(), dilations.size(),
        platform::errors::InvalidArgument(
            "The dims of strides should be the same with that of dilations. "
            "But recieved dims(strides: %u) != dims(dilations: %u).",
            strides.size(), dilations.size()));

    // check kernel_sizes
    PADDLE_ENFORCE_GT(kernel_sizes[0], 0,
                      platform::errors::InvalidArgument(
                          "The `kernel_sizes` should be greater than zero, "
                          "but recieved kernel_height: %d kernel_width: %d.",
                          kernel_sizes[0], kernel_sizes[1]));
    PADDLE_ENFORCE_GT(kernel_sizes[1], 0,
                      platform::errors::InvalidArgument(
                          "The `kernel_sizes` should be greater than zero, "
                          "but recieved kernel_height: %d kernel_width: %d.",
                          kernel_sizes[0], kernel_sizes[1]));
    // check strides
    PADDLE_ENFORCE_GT(strides[0], 0,
                      platform::errors::InvalidArgument(
                          "The `strides` should be greater than zero, "
                          "but recieved strides_height: %d strides_width: %d.",
                          strides[0], strides[1]));
    PADDLE_ENFORCE_GT(strides[1], 0,
                      platform::errors::InvalidArgument(
                          "The `strides` should be greater than zero, "
                          "but recieved strides_height: %d strides_width: %d.",
                          strides[0], strides[1]));
    // check dilations
    PADDLE_ENFORCE_GT(
        dilations[0], 0,
        platform::errors::InvalidArgument(
            "The `dilations` should be greater than zero, "
            "but recieved dilations_height: %d dilations_width: %d.",
            dilations[0], dilations[1]));
    PADDLE_ENFORCE_GT(
        dilations[1], 0,
        platform::errors::InvalidArgument(
            "The `dilations` should be greater than zero, "
            "but recieved dilations_height: %d dilations_width: %d.",
            dilations[0], dilations[1]));

    std::vector<int> out_dims;
    out_dims.push_back(in_dims[0]);
    int output_channels = in_dims[1] * kernel_sizes[0] * kernel_sizes[1];
    out_dims.push_back(output_channels);

    int output_height =
        CalcOutputSize(in_dims[2], kernel_sizes[0], dilations[0], paddings[0],
                       paddings[2], strides[0]);
    int output_width = CalcOutputSize(in_dims[3], kernel_sizes[1], dilations[1],
                                      paddings[1], paddings[3], strides[1]);
    if (ctx->IsRuntime()) {
      // only check output height and width in runtime
      PADDLE_ENFORCE_GT(
          output_height, 0,
          platform::errors::InvalidArgument(
              "The sliding blocks calculated from input spatial size "
              "(%d, %d), kernel_sizes (%d, %d), strides (%d, %d), "
              "dilations (%d, %d), is (%d, %d), which should be a "
              "positive integer.",
              in_dims[2], in_dims[3], kernel_sizes[0], kernel_sizes[1],
              strides[0], strides[1], dilations[0], dilations[1], output_height,
              output_width));
      PADDLE_ENFORCE_GT(
          output_width, 0,
          platform::errors::InvalidArgument(
              "The sliding blocks calculated from input spatial size "
              "(%d, %d), kernel_sizes (%d, %d), strides (%d, %d), "
              "dilations (%d, %d), is (%d, %d), which should be a "
              "positive integer.",
              in_dims[2], in_dims[3], kernel_sizes[0], kernel_sizes[1],
              strides[0], strides[1], dilations[0], dilations[1], output_height,
              output_width));
    }
    int output_col_length = output_height * output_width;
    out_dims.push_back(output_col_length);
    ctx->SetOutputDim("Y", framework::make_ddim(out_dims));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class UnfoldGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Y")), true,
        platform::errors::NotFound("The gradient of Y should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("The input X should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::NotFound("The gradient of X should not be null"));
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Y")),
                                   ctx.device_context());
  }
};

template <typename T>
class UnfoldGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("unfold_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetInput("X", this->Input("X"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(UnfoldGradOpNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(unfold, ops::UnfoldOp, ops::UnfoldOpMaker,
                  ops::UnfoldGradMaker<paddle::framework::OpDesc>,
                  ops::UnfoldGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(unfold_grad, ops::UnfoldGradOp,
                  ops::UnfoldGradOpNoNeedBufferVarsInferer);

REGISTER_OP_CPU_KERNEL(
    unfold, ops::UnfoldOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::UnfoldOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    unfold_grad,
    ops::UnfoldGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::UnfoldGradOpKernel<paddle::platform::CPUDeviceContext, double>);
