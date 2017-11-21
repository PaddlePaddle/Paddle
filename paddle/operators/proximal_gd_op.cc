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

#include "paddle/operators/proximal_gd_op.h"

namespace paddle {
namespace operators {

class ProximalGDOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of ProximalGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of ProximalGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of ProximalGDOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of ProximalGDOp should not be null.");

    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(param_dim, ctx->GetInputDim("Grad"),
                      "Two input of ProximalGD Op's dimension must be same.");

    auto lr_dim = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(framework::product(lr_dim), 1,
                      "Learning Rate should be a scalar.");

    ctx->SetOutputDim("ParamOut", param_dim);
  }
};

class ProximalGDOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ProximalGDOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Param",
             "(Tensor, default Tensor<float>) "
             "Input parameter value that has to be updated.");
    AddInput("Grad",
             "(Tensor, default Tensor<float>) "
             "Input gradient of the parameter.");
    AddInput("LearningRate",
             "(Tensor, default Tensor<float>) "
             "The learning rate should be a tensor of size 1.");

    AddOutput("ParamOut", "(Tensor) Output updated parameter value.");

    AddAttr<float>("l1",
                   "(float, default 0.0) "
                   "L1 regularization strength.")
        .SetDefault(0.0f);
    AddAttr<float>("l2",
                   "(float, default 0.0) "
                   "L2 regularization strength.")
        .SetDefault(0.0f);
    AddComment(R"DOC(
ProximalGD Operator.

Optimizer that implements the proximal gradient descent algorithm:

$$
prox\_param = param - learning\_rate * grad \\
param = sign(prox\_param) / (1 + learning\_rate * l2) *
        \max(|prox\_param| - learning\_rate * l1, 0)
$$        

The paper that proposed Proximal Gradient Descent:
(http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf)

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(proximal_gd, ops::ProximalGDOp,
                             ops::ProximalGDOpMaker);
REGISTER_OP_CPU_KERNEL(
    proximal_gd, ops::ProximalGDOpKernel<paddle::platform::CPUPlace, float>);
