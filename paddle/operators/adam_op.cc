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

#include "paddle/operators/adam_op.h"

namespace paddle {
namespace operators {

class AdamOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Moment1"),
                   "Input(Moment1) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Moment2"),
                   "Input(Moment2) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Beta1Pow"),
                   "Input(Beta1Pow) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Beta2Pow"),
                   "Input(Beta2Pow) of AdamOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Moment1Out"),
                   "Output(Moment1Out) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Moment2Out"),
                   "Output(Moment2Out) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Beta1PowOut"),
                   "Output(Beta1PowOut) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Beta2PowOut"),
                   "Output(Beta2PowOut) of AdamOp should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "Learning rate should have 1 dimension");
    auto beta1_pow_dims = ctx->GetInputDim("Beta1Pow");
    PADDLE_ENFORCE_EQ(framework::product(beta1_pow_dims), 1,
                      "Beta1 power accumulator should have 1 dimension");
    auto beta2_pow_dims = ctx->GetInputDim("Beta2Pow");
    PADDLE_ENFORCE_EQ(framework::product(beta1_pow_dims), 1,
                      "Beta1 power accumulator should have 1 dimension");

    auto param_dims = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Grad"),
        "Param and Grad input of AdamOp should have same dimension");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Moment1"),
        "Param and Moment input of AdamOp should have same dimension");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Moment2"),
        "Param and InfNorm input of AdamOp should have same dimension");

    ctx->SetOutputDim("ParamOut", param_dims);
    ctx->SetOutputDim("Moment1Out", param_dims);
    ctx->SetOutputDim("Moment2Out", param_dims);
    ctx->SetOutputDim("Beta1PowOut", beta1_pow_dims);
    ctx->SetOutputDim("Beta2PowOut", beta2_pow_dims);
  }
};

class AdamOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AdamOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("LearningRate", "(Tensor) Learning rate");
    AddInput("Moment1", "(Tensor) Input first moment");
    AddInput("Moment2", "(Tensor) Input second moment");
    AddInput("Beta1Pow", "(Tensor) Input beta1 power accumulator");
    AddInput("Beta2Pow", "(Tensor) Input beta2 power accumulator");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("Moment1Out", "(Tensor) Output first moment");
    AddOutput("Moment2Out", "(Tensor) Output second moment");
    AddOutput("Beta1PowOut", "(Tensor) Output beta1 power accumulator");
    AddOutput("Beta2PowOut", "(Tensor) Output beta2 power accumulator");

    AddAttr<float>("beta1",
                   "(float, default 0.9) "
                   "Exponential decay rate for the "
                   "first moment estimates.")
        .SetDefault(0.9f);
    AddAttr<float>("beta2",
                   "(float, default 0.999) "
                   "exponential decay rate for the "
                   "second moment estimates.")
        .SetDefault(0.999f);
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-8) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-8f);

    AddComment(R"DOC(
Adam Updates Operator.

This implements the Adam optimizer from Section 2 of the Adam
paper[1]. Adam is a first-order gradient-based optimization
method based on adaptive estimates of lower-order moments.

Adam updates:

moment1_out = beta1 * moment1 + (1 − beta1) * grad
moment2_out = beta2 * moment2 + (1 − beta2) * grad * grad
beta1_pow_out = beta1_pow * beta1
beta2_pow_out = beta2_pow * beta2
learning_rate_t = learning_rate_t *
                  sqrt(1 - beta2_pow_out) / (1 - beta1_pow_out)
param_out = param - learning_rate_t * moment1/ (sqrt(moment2) + epsilon)

References:
  [1] Adam: A Method for Stochastic Optimization
      (https://arxiv.org/abs/1412.6980)

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adam, ops::AdamOp, ops::AdamOpMaker);
REGISTER_OP_CPU_KERNEL(adam,
                       ops::AdamOpKernel<paddle::platform::CPUPlace, float>);
