/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fsp_op.h"

namespace paddle {
namespace operators {

class FSPOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of FSPOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) of FSPOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FSPOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE(
        x_dims.size() == 4,
        "The Input(X) must have shape [batch_size, channel, height, width].");
    PADDLE_ENFORCE(
        y_dims.size() == 4,
        "The Input(Y) must have shape [batch_size, channel, height, width].");
    PADDLE_ENFORCE(
        (x_dims[2] == y_dims[2]) && (x_dims[3] == y_dims[3]),
        "The Input(X) and Input(Y) should have the same height and width.");

    ctx->SetOutputDim("Out", {x_dims[0], x_dims[1], y_dims[1]});
    ctx->ShareLoD("X", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
    framework::DataLayout layout_ = framework::DataLayout::kAnyLayout;
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context(), layout_, library_);
  }
};

class FSPOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The input of FSP op with shape [batch_size, x_channel, "
             "weight, height]");
    AddInput("Y",
             "(Tensor) The input of FSP op with shape"
             "[batch_size, y_channel, weight, height]."
             "The y_channel can be different with the x_channel of Input(X)"
             " while the other dimensions must be the same with Input(X)'s.");
    AddOutput(
        "Out",
        "(Tensor) The output of FSP op with shape "
        "[batch_size, x_channel, y_channel]. The x_channel is the channel "
        "of Input(X) and the y_channel is the channel of Input(Y).");
    AddComment(R"DOC(
    This op is used to calculate the flow of solution procedure (FSP) matrix of two feature maps.
    Given feature map x with shape [x_channel, w, h] and feature map y with shape
    [y_channel, w, h], we can get the fsp matrix of x and y in two steps:

        step 1: reshape x into matrix with shape [x_channel, w * h] and reshape and
                transpose y into matrix with shape [w * h, y_channel]
        step 2: multiply x and y to get fsp matrix with shape [x_channel, y_channel]

    The output is a batch of fsp matrices.
    )DOC");
  }
};

class FSPOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"))->type(),
        ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fsp, ops::FSPOp, ops::FSPOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(fsp_grad, ops::FSPOpGrad);
REGISTER_OP_CPU_KERNEL(
    fsp, ops::FSPOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FSPOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    fsp_grad, ops::FSPGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FSPGradOpKernel<paddle::platform::CPUDeviceContext, double>);
