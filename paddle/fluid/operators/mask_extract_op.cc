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

#include "paddle/fluid/operators/mask_extract_op.h"

namespace paddle {
namespace operators {

using framework::LoDTensor;

class MaskExtractOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of MaskExtractOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Mask"),
                   "Input(Mask) of MaskExtractOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MaskExtractOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Ids"),
                   "Output(Ids) of MaskExtractOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Offset"),
                   "Output(Offset) of MaskExtractOp should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    auto mask_dims = ctx->GetInputDim("Mask");
    PADDLE_ENFORCE(mask_dims.size() == 2 && mask_dims[1] == 1,
                   "Input(Mask) should have the shape like [batch_size, 1].");
    PADDLE_ENFORCE_EQ(x_dims[0], mask_dims[0], "Input(X) and Input(Mask) "
                   "should have the same first dimension.");
    ctx->SetOutputDim("Out", x_dims);
    ctx->SetOutputDim("Ids", mask_dims);
    ctx->SetOutputDim("Offset", mask_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
        ctx.GetPlace());
  }

};

class MaskExtractOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, default LoDTensor<float>) Input data.");
    AddInput("Mask",
             "(LoDTensor, default LoDTensor<float>) Mask information");
    AddOutput("Out",
              "(LodTensor, default LoDTensor<float>). ");
    AddOutput("Ids",
              "(LodTensor, default LoDTensor<float>). ");
    AddOutput("Offset",
              "(LodTensor, default LoDTensor<float>). ").AsIntermediate();
    AddComment(R"DOC(
Mask Extract Operator.

This operator extracts unmasked data from input, with the given mask.

)DOC");
  }
};

class MaskExtractOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Offset"), "Input(Out) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
        ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mask_extract, ops::MaskExtractOp,
                  ops::MaskExtractOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(mask_extract_grad, ops::MaskExtractOpGrad);
REGISTER_OP_CPU_KERNEL(
    mask_extract,
    ops::MaskExtractCPUKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MaskExtractCPUKernel<paddle::platform::CPUDeviceContext, double>,
    ops::MaskExtractCPUKernel<paddle::platform::CPUDeviceContext, int>,
    ops::MaskExtractCPUKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    mask_extract_grad,
    ops::MaskExtractCPUGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MaskExtractCPUGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::MaskExtractCPUGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::MaskExtractCPUGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
