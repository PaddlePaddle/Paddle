/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/adamax_op.h"

namespace paddle {
namespace operators {

class AdamaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Moment"),
                   "Input(Moment) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("InfNorm"),
                   "Input(InfNorm) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Beta1Pow"),
                   "Input(Beta1Pow) of AdamaxOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("MomentOut"),
                   "Output(MomentOut) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("InfNormOut"),
                   "Output(InfNormOut) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Beta1PowOut"),
                   "Output(Beta1PowOut) of AdamaxOp should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "Learning rate should have 1 dimension");
    auto beta1_pow_dims = ctx->GetInputDim("Beta1Pow");
    PADDLE_ENFORCE_EQ(framework::product(beta1_pow_dims), 1,
                      "Beta1 power accumulator should have 1 dimension");
    auto param_dims = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Grad"),
        "Param and Grad input of AdamaxOp should have same dimension");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Moment"),
        "Param and Moment input of AdamaxOp should have same dimension");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("InfNorm"),
        "Param and InfNorm input of AdamaxOp should have same dimension");

    ctx->SetOutputDim("ParamOut", param_dims);
    ctx->SetOutputDim("MomentOut", param_dims);
    ctx->SetOutputDim("InfNormOut", param_dims);
    ctx->SetOutputDim("Beta1PowOut", beta1_pow_dims);
  }
};

class AdamaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AdamaxOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
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
    AddOutput("Beta1PowOut", "(Tensor) Output beta1 power accumulator");

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
Adamax Updates Operator.

This implements the Adamax optimizer from Section 7 of the Adam
paper[1]. Adamax is a variant of the
Adam algorithm based on the infinity norm.

Adamax updates:

moment_out = beta1 * moment + (1 - beta1) * grad
inf_norm_out = max(beta2 * inf_norm + epsilon, abs(grad))
beta1_pow_out = beta1_pow * beta1
learning_rate_t = learning_rate/(1 - beta1_pow_out)
param_out = param - learning_rate_t * moment_out/inf_norm_out

The original paper does not have an epsilon attribute.
However, it is added here for numerical stability
by preventing divide by 0.

References:
  [1] Adam: A Method for Stochastic Optimization
      (https://arxiv.org/abs/1412.6980)

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adamax, ops::AdamaxOp, ops::AdamaxOpMaker);
REGISTER_OP_CPU_KERNEL(adamax,
                       ops::AdamaxOpKernel<paddle::platform::CPUPlace, float>);
