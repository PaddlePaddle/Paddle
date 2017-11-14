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

#include "paddle/operators/momentum_op.h"

namespace paddle {
namespace operators {

class MomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(param) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(grad) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Velocity"),
                   "Input(velocity) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of Momentum should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("VelocityOut"),
                   "Output(VelocityOut) of Momentum should not be null.");

    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("Grad"),
        "Param and Grad input of MomentumOp should have the same dimension.");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("Velocity"),
        "Param and Velocity of MomentumOp should have the same dimension.");
    PADDLE_ENFORCE_EQ(framework::product(ctx->GetInputDim("LearningRate")), 1,
                      "Learning_rate should be a scalar");

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("VelocityOut", param_dim);
  }
};

class MomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MomentumOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Param",
             "(Tensor, default Tensor<float>) "
             "Input parameter that has to be updated");
    AddInput("Grad",
             "(Tensor, default Tensor<float>) "
             "Input gradient of the parameter");
    AddInput("Velocity",
             "(Tensor, default Tensor<float>) "
             "Input velocity (corresponding to the parameter) "
             "that has to be updated");
    AddInput("LearningRate",
             "(Tensor, default Tensor<float>) "
             "Input learning rate");

    AddOutput("ParamOut", "(Tensor) Output updated parameter");
    AddOutput("VelocityOut", "(Tensor) Output updated velocity");

    AddAttr<float>("mu", "(float) Momentum coefficient");
    AddAttr<bool>("use_nesterov",
                  "(bool, default false) "
                  "Use Nesterov Momentum")
        .SetDefault(false);
    AddComment(R"DOC(
Momentum Optimizer.

This optimizer has a flag for Nestrov Momentum.
The update equations are as follows:

$$
velocity = mu * velocity + gradient \\
if (use\_nesterov):   \\
  param = param - gradient * learning\_rate + mu * velocity * learning\_rate \\
else:   \\
  param = param - learning\_rate * velocity. \\
$$

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(momentum, ops::MomentumOp, ops::MomentumOpMaker);
REGISTER_OP_CPU_KERNEL(
    momentum, ops::MomentumOpKernel<paddle::platform::CPUPlace, float>);
