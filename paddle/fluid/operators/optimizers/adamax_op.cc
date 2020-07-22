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

#include "paddle/fluid/operators/optimizers/adamax_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class AdamaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Param"), "Input", "Param", "Adamax");
    OP_INOUT_CHECK(ctx->HasInput("Grad"), "Input", "Grad", "Adamax");
    OP_INOUT_CHECK(ctx->HasInput("Moment"), "Input", "Moment", "Adamax");
    OP_INOUT_CHECK(ctx->HasInput("InfNorm"), "Input", "InfNorm", "Adamax");
    OP_INOUT_CHECK(ctx->HasInput("LearningRate"), "Input", "LearningRate",
                   "Adamax");
    OP_INOUT_CHECK(ctx->HasInput("Beta1Pow"), "Input", "Beta1Pow", "Adamax");
    PADDLE_ENFORCE_EQ(
        ctx->GetInputsVarType("Param").front(),
        framework::proto::VarType::LOD_TENSOR,
        platform::errors::InvalidArgument(
            "The input var's type should be LoDTensor, but the received is %s",
            ctx->Inputs("Param").front(),
            ctx->GetInputsVarType("Param").front()));
    PADDLE_ENFORCE_EQ(
        ctx->GetInputsVarType("Grad").front(),
        framework::proto::VarType::LOD_TENSOR,
        platform::errors::InvalidArgument(
            "The input var's type should be LoDTensor, but the received is %s",
            ctx->Inputs("Grad").front(),
            ctx->GetInputsVarType("Grad").front()));

    OP_INOUT_CHECK(ctx->HasOutput("ParamOut"), "Output", "ParamOut", "Adamax");
    OP_INOUT_CHECK(ctx->HasOutput("MomentOut"), "Output", "MomentOut",
                   "Adamax");
    OP_INOUT_CHECK(ctx->HasOutput("InfNormOut"), "Output", "InfNormOut",
                   "Adamax");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(framework::product(lr_dims), 0,
                      platform::errors::InvalidArgument(
                          "Maybe the Input variable LearningRate has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function."));
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      platform::errors::InvalidArgument(
                          "Learning rate should have 1 dimension"));
    auto beta1_pow_dims = ctx->GetInputDim("Beta1Pow");
    PADDLE_ENFORCE_EQ(framework::product(beta1_pow_dims), 1,
                      platform::errors::InvalidArgument(
                          "Beta1 power accumulator should have 1 dimension"));
    auto param_dims = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Grad"),
        platform::errors::InvalidArgument(
            "Param and Grad input of AdamaxOp should have same dimension"));
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Moment"),
        platform::errors::InvalidArgument(
            "Param and Moment input of AdamaxOp should have same dimension"));
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("InfNorm"),
        platform::errors::InvalidArgument(
            "Param and InfNorm input of AdamaxOp should have same dimension"));

    ctx->SetOutputDim("ParamOut", param_dims);
    ctx->SetOutputDim("MomentOut", param_dims);
    ctx->SetOutputDim("InfNormOut", param_dims);
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Param"), ctx.GetPlace());
  }
};

class AdamaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("LearningRate", "(Tensor) Learning rate");
    AddInput("Moment", "(Tensor) First moment");
    AddInput("InfNorm",
             "(Tensor) "
             "Input exponentially weighted infinity norm");
    AddInput("Beta1Pow", "(Tensor) Input beta1 power accumulator");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("MomentOut", "(Tensor) Output first moment");
    AddOutput("InfNormOut",
              "(Tensor) "
              "Output exponentially weighted infinity norm");

    AddAttr<float>("beta1",
                   "(float, default 0.9) "
                   "Exponential decay rate for the "
                   "1st moment estimates.")
        .SetDefault(0.9f);
    AddAttr<float>("beta2",
                   "(float, default 0.999) "
                   "exponential decay rate for the weighted "
                   "infinity norm estimates.")
        .SetDefault(0.999f);
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-8) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-8f);
    AddComment(R"DOC(
Adamax Optimizer.

We implement the Adamax optimizer from Section 7 of the Adam
paper: https://arxiv.org/abs/1412.6980. Adamax is a variant of the
Adam algorithm based on the infinity norm.

Adamax updates:

$$
moment\_out = \beta_1 * moment + (1 - \beta_1) * grad \\
inf\_norm\_out = max(\beta_2 * inf\_norm + \epsilon, |grad|) \\
learning\_rate = \frac{learning\_rate}{1 - \beta_{1\_pow}} \\
param\_out = param - learning\_rate * \frac{moment\_out}{inf\_norm\_out}
$$

The original paper does not have an epsilon attribute.
However, it is added here for numerical stability to prevent the
division by 0 error.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adamax, ops::AdamaxOp, ops::AdamaxOpMaker);
REGISTER_OP_CPU_KERNEL(
    adamax, ops::AdamaxOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AdamaxOpKernel<paddle::platform::CPUDeviceContext, double>);
