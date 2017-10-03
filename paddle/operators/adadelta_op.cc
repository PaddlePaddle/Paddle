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

#include "paddle/operators/adadelta_op.h"

namespace paddle {
namespace operators {

class AdadeltaOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("param"),
                   "Input(param) of AdadeltaOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("grad"),
                   "Input(grad) of AdadeltaOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("avg_squared_grad"),
                   "Input(avg_squared_grad) of AdadeltaOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("avg_squared_update"),
        "Input(avg_squared_update) of AdadeltaOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("param_out"),
                   "Output(param_out) of AdadeltaOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("avg_squared_grad_out"),
        "Output(avg_squared_grad_out) of AdadeltaOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("avg_squared_update_out"),
        "Output(avg_squared_update_out) of AdadeltaOp should not be null.");

    auto param_dim = ctx->GetInputDim("param");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("grad"),
        "param and grad input of AdadeltaOp should have same dimension");
    PADDLE_ENFORCE_EQ(param_dim, ctx->GetInputDim("avg_squared_grad"),
                      "param and avg_squared_grad input of AdadeltaOp "
                      "should have same dimension");
    PADDLE_ENFORCE_EQ(param_dim, ctx->GetInputDim("avg_squared_update"),
                      "param and avg_squared_update input of AdadeltaOp "
                      "should have same dimension");

    ctx->SetOutputDim("param_out", param_dim);
    ctx->SetOutputDim("avg_squared_grad_out", param_dim);
    ctx->SetOutputDim("avg_squared_update_out", param_dim);
  }
};

class AdadeltaOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AdadeltaOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("param", "Input parameter");
    AddInput("grad", "Input gradient");
    AddInput("avg_squared_grad", "Input expectation of squared gradient");
    AddInput("avg_squared_update",
             "Input expectation of squared parameter updates");

    AddOutput("param_out", "Output parameter");
    AddOutput("avg_squared_grad_out", "Output expectation of squared gradient");
    AddOutput("avg_squared_update_out",
              "Output expectation of squared parameter updates");

    AddAttr<float>("rho", "exponential decay rate for squared gradients.");
    AddAttr<float>("epsilon", "Constant for numerical stability");
    AddComment(R"DOC(
Adadelta Updates Operator.

This implements the Adadelta optimizer from https://arxiv.org/abs/1212.5701.
Adadelta is a per0-dimension learning rate method for gradient descent.

Adadelta updates:

avg_squared_grad_out = rho * avg_squared_grad + (1 - rho) * grad * grad
param_update =  - sqrt((avg_squared_update + epsilon) /
                       (avg_squared_grad_out + epsilon)) * grad
avg_squared_update_out = rho * avg_squared_update + (1 - rho) * param_update**2
param_out = param + param_update

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adadelta, ops::AdadeltaOp, ops::AdadeltaOpMaker);
REGISTER_OP_CPU_KERNEL(
    adadelta, ops::AdadeltaOpKernel<paddle::platform::CPUPlace, float>);
