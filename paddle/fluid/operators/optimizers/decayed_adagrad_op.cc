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

#include "paddle/fluid/operators/optimizers/decayed_adagrad_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class DecayedAdagradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of DecayedAdagradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of DecayedAdagradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Moment"),
                   "Input(Moment) of DecayedAdagradOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("LearningRate"),
        "Input(LearningRate) of DecayedAdagradOp should not be null.");
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
                   "Output(ParamOut) of DecayedAdagradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("MomentOut"),
                   "Output(MomentOut) of DecayedAdagradOp should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(framework::product(lr_dims), 0,
                      "Maybe the Input variable LearningRate has not "
                      "been initialized. You may need to confirm "
                      "if you put exe.run(startup_program) "
                      "after optimizer.minimize function.");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "LearningRate should have one element");
    auto param_dims = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(param_dims, ctx->GetInputDim("Grad"),
                      "Param and Grad input of DecayedAdagradOp should have "
                      "the same dimension.");
    PADDLE_ENFORCE_EQ(param_dims, ctx->GetInputDim("Moment"),
                      "Param and Moment input of DecayedAdagradOp should have "
                      "the same dimension.");

    ctx->SetOutputDim("ParamOut", param_dims);
    ctx->SetOutputDim("MomentOut", param_dims);
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Param")->type(),
                                   ctx.GetPlace());
  }
};

class DecayedAdagradOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("Moment", "(Tensor) Second moment");
    AddInput("LearningRate", "(Tensor) Learning rate");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("MomentOut", "(Tensor) Output second moment");

    AddAttr<float>("decay",
                   "(float, default 0.95) "
                   "Discounting factor for coming gradient")
        .SetDefault(0.95);
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-6) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-6f);
    AddComment(R"DOC(
Decayed Adagrad Optimizer.

The update is done as follows:

$$
moment\_out = decay * moment + (1 - decay) * grad * grad \\
param\_out = param - \frac{learning\_rate * grad}{\sqrt{moment\_out} + epsilon}
$$

The original paper(http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
does not have an epsilon attribute. It is added here for numerical
stability to avoid the division by zero error.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(decayed_adagrad, ops::DecayedAdagradOp,
                             ops::DecayedAdagradOpMaker);
REGISTER_OP_CPU_KERNEL(
    decayed_adagrad,
    ops::DecayedAdagradOpKernel<paddle::platform::CPUDeviceContext, float>);
