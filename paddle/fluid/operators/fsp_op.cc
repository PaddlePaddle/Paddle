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
    AddInput("X", "(Tensor) The input of FSP op");
    AddInput("Y", "(Tensor) The input of FSP op");
    AddOutput("Out", "(Tensor) The output of FSP op");
    AddComment(R"DOC(
    Calculate FSP matrix.
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
    fsp, ops::FSPOpKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    fsp_grad, ops::FSPGradOpKernel<paddle::platform::CPUDeviceContext, float>);
