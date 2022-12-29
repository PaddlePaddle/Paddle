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

#include "paddle/fluid/operators/optimizers/proximal_gd_op.h"

namespace paddle {
namespace operators {

class ProximalGDOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Param"), "Input", "Param", "ProximalGDOp");
    OP_INOUT_CHECK(ctx->HasInput("Grad"), "Input", "Grad", "ProximalGDOp");
    OP_INOUT_CHECK(
        ctx->HasInput("LearningRate"), "Input", "LearningRate", "ProximalGDOp");

    OP_INOUT_CHECK(
        ctx->HasOutput("ParamOut"), "Output", "Paramout", "ProximalGDOp");

    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(param_dim,
                      ctx->GetInputDim("Grad"),
                      platform::errors::InvalidArgument(
                          "The shape of Intput(Param) should be equal to the "
                          "Input(Grad) of ProximalGD Op. But received "
                          "Input(Param).dimensions=[%s], "
                          "Input(Grad).dimensions=[%s]",
                          param_dim,
                          ctx->GetInputDim("Grad")));

    auto lr_dim = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(
        phi::product(lr_dim),
        1,
        platform::errors::InvalidArgument(
            "Learning Rate should be a scalar. But received dimmensions:[%s]",
            lr_dim));

    ctx->SetOutputDim("ParamOut", param_dim);
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Param"), ctx.GetPlace());
  }
};

class ProximalGDOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
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
REGISTER_OP_WITHOUT_GRADIENT(proximal_gd,
                             ops::ProximalGDOp,
                             ops::ProximalGDOpMaker);
REGISTER_OP_CPU_KERNEL(proximal_gd,
                       ops::ProximalGDOpKernel<phi::CPUContext, float>);
