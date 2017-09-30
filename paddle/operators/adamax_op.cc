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
  void InferShape(framework::InferShapeContextBase *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("param"),
                   "Input(param) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("grad"),
                   "Input(grad) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("moment"),
                   "Input(moment) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("inf_norm"),
                   "Input(inf_norm) of AdamaxOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("param_out"),
                   "Output(param_out) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("moment_out"),
                   "Output(moment_out) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("inf_norm_out"),
                   "Output(inf_norm_out) of AdamaxOp should not be null.");

    auto param_dim = ctx->GetInputDim("param");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("grad"),
        "param and grad input of AdamaxOp should have same dimension");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("moment"),
        "param and moment input of AdamaxOp should have same dimension");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("inf_norm"),
        "param and inf_norm input of AdamaxOp should have same dimension");

    ctx->SetOutputDim("param_out", param_dim);
    ctx->SetOutputDim("moment_out", param_dim);
    ctx->SetOutputDim("inf_norm_out", param_dim);
  }
};

class AdamaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AdamaxOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("param", "Input parameter");
    AddInput("grad", "Input gradient");
    AddInput("moment", "First moment");
    AddInput("inf_norm", "Input exponentially weighted infinity norm");

    AddOutput("param_out", "Output parameter");
    AddOutput("moment_out", "Output first moment");
    AddOutput("inf_norm_out", "Output exponentially weighted infinity norm");

    AddAttr<int>("time_step", "Time step");
    AddAttr<float>("learning_rate", "Learning rate");
    AddAttr<float>("beta_1",
                   "exponential decay rate for the 1st moment estimates.");
    AddAttr<float>(
        "beta_2",
        "exponential decay rate for the weighted infinity norm estimates.");
    AddAttr<float>("epsilon", "Constant for numerical stability");
    AddComment(R"DOC(
Adamax Updates Operator.

This implements the Adamax optimizer from Section 7 of the Adam
paper(https://arxiv.org/abs/1412.6980). Adamax is a variant of the
Adam algorithm based on the infinity norm.

Adamax updates:

moment_out = beta_1 * moment + (1 - beta_1) * grad
inf_norm_out = max(beta_2 * inf_norm + epsilon, abs(grad))
param_out = param - (learning_rate/(1 - beta_1^t)) * moment_out/inf_norm_out

The original paper(https://arxiv.org/abs/1412.6980) does not have an
epsilon attribute. However, it is added here for numerical stability
by preventing divide by 0.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adamax, ops::AdamaxOp, ops::AdamaxOpMaker);
REGISTER_OP_CPU_KERNEL(adamax,
                       ops::AdamaxOpKernel<paddle::platform::CPUPlace, float>);
