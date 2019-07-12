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

#include "paddle/fluid/operators/optimizers/adadelta_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class AdadeltaOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of AdadeltaOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of AdadeltaOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("AvgSquaredGrad"),
                   "Input(AvgSquaredGrad) of AdadeltaOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("AvgSquaredUpdate"),
                   "Input(AvgSquaredUpdate) of AdadeltaOp should not be null.");
    PADDLE_ENFORCE(
        ctx->GetInputsVarType("Param").front() ==
            framework::proto::VarType::LOD_TENSOR,
        "The input var's type should be LoDTensor, but the received is %s",
        ctx->Inputs("Param").front(), ctx->GetInputsVarType("Param").front());
    PADDLE_ENFORCE(
        ctx->GetInputsVarType("Grad").front() ==
            framework::proto::VarType::LOD_TENSOR,
        "The input var's type should be LoDTensor, but the received is %s",
        ctx->Inputs("Grad").front(), ctx->GetInputsVarType("Grad").front());

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of AdadeltaOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("AvgSquaredGradOut"),
        "Output(AvgSquaredGradOut) of AdadeltaOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("AvgSquaredUpdateOut"),
        "Output(AvgSquaredUpdateOut) of AdadeltaOp should not be null.");

    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("Grad"),
        "param and grad input of AdadeltaOp should have same dimension");
    PADDLE_ENFORCE_EQ(param_dim, ctx->GetInputDim("AvgSquaredGrad"),
                      "Param and AvgSquaredGrad input of AdadeltaOp "
                      "should have same dimension");
    PADDLE_ENFORCE_EQ(param_dim, ctx->GetInputDim("AvgSquaredUpdate"),
                      "Param and AvgSquaredUpdate input of AdadeltaOp "
                      "should have same dimension");

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("AvgSquaredGradOut", param_dim);
    ctx->SetOutputDim("AvgSquaredUpdateOut", param_dim);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Param")->type(),
                                   ctx.GetPlace());
  }
};

class AdadeltaOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("AvgSquaredGrad", "(Tensor) Input average of squared gradient");
    AddInput("AvgSquaredUpdate",
             "(Tensor) Input average of squared parameter updates");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("AvgSquaredGradOut",
              "(Tensor) Output average of squared gradient");
    AddOutput("AvgSquaredUpdateOut",
              "(Tensor) Output average of squared parameter updates");

    AddAttr<float>("rho",
                   "(float, default 0.95) Exponential decay rate "
                   "for squared gradients.")
        .SetDefault(0.95f);
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-6) Constant for "
                   "numerical stability")
        .SetDefault(1.0e-6f);
    AddComment(R"DOC(
Adadelta Optimizer.

Adadelta optimizer is implemented as explained in:
https://arxiv.org/abs/1212.5701
Adadelta is a per-dimension adaptive learning rate method used
for gradient descent.

Adadelta updates are as follows:

$$
avg\_squared\_grad\_out = \rho * avg\_squared\_grad + (1 - \rho) * grad * grad \\
param\_update =  - \sqrt{\frac{avg\_squared\_update + \epsilon}{avg\_squared\_grad\_out + \epsilon}} * grad \\
avg\_squared\_update\_out = \rho * avg\_squared\_update + (1 - \rho) * {param\_update}^2 \\
param\_out = param + param\_update
$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adadelta, ops::AdadeltaOp, ops::AdadeltaOpMaker);
REGISTER_OP_CPU_KERNEL(
    adadelta, ops::AdadeltaOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AdadeltaOpKernel<paddle::platform::CPUDeviceContext, double>);
