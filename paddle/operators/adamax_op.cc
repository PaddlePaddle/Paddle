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
    PADDLE_ENFORCE(ctx->HasInput("TimeStep"),
                   "Input(TimeStep) of AdamaxOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("MomentOut"),
                   "Output(MomentOut) of AdamaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("InfNormOut"),
                   "Output(InfNormOut) of AdamaxOp should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "Learning rate should have 1 dimension");
    auto t_dims = ctx->GetInputDim("TimeStep");
    PADDLE_ENFORCE_EQ(framework::product(t_dims), 1,
                      "Time step should have 1 dimension");
    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("Grad"),
        "Param and Grad input of AdamaxOp should have same dimension");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("Moment"),
        "Param and Moment input of AdamaxOp should have same dimension");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("InfNorm"),
        "Param and InfNorm input of AdamaxOp should have same dimension");

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("MomentOut", param_dim);
    ctx->SetOutputDim("InfNormOut", param_dim);
  }

  // Datatype of operator is determined by Param tensor
  framework::DataType IndicateDataType(
      const framework::ExecutionContext &ctx) const override {
    return framework::ToDataType(ctx.Input<Tensor>("Param")->type());
  }
};

class AdamaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AdamaxOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Param", "(Tensor, default Tensor<float>) Input parameter");
    AddInput("Grad", "(Tensor, default Tensor<float>) Input gradient");
    AddInput("LearningRate", "(Tensor, default Tensor<float>) Learning rate");
    AddInput("Moment", "(Tensor, default Tensor<float>) First moment");
    AddInput("InfNorm",
             "(Tensor, default Tensor<float>) "
             "Input exponentially weighted infinity norm");
    AddInput("TimeStep", "(Tensor, default Tensor<int>) Time step");

    AddOutput("ParamOut", "(Tensor, default Tensor<float>) Output parameter");
    AddOutput("MomentOut",
              "(Tensor, default Tensor<float>) Output first moment");
    AddOutput("InfNormOut",
              "(Tensor, default Tensor<float>) "
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
Adamax Updates Operator.

This implements the Adamax optimizer from Section 7 of the Adam
paper[1]. Adamax is a variant of the
Adam algorithm based on the infinity norm.

Adamax updates:

moment_out = beta_1 * moment + (1 - beta_1) * grad
inf_norm_out = max(beta_2 * inf_norm + epsilon, abs(grad))
param_out = param - (learning_rate/(1 - beta_1^t)) * moment_out/inf_norm_out

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
