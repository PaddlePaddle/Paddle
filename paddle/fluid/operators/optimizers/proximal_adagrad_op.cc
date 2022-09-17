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

#include "paddle/fluid/operators/optimizers/proximal_adagrad_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class ProximalAdagradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Param"), "Input", "Param", "ProximalAdagradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Moment"), "Input", "Moment", "ProximalAdagradOp");
    OP_INOUT_CHECK(ctx->HasInput("Grad"), "Input", "Grad", "ProximalAdagradOp");
    OP_INOUT_CHECK(ctx->HasInput("LearningRate"),
                   "Input",
                   "LearningRate",
                   "ProximalAdagradOp");

    OP_INOUT_CHECK(
        ctx->HasOutput("ParamOut"), "Output", "ParamOut", "ProximalAdagradOp");
    OP_INOUT_CHECK(ctx->HasOutput("MomentOut"),
                   "Output",
                   "MomentOut",
                   "ProximalAdagradOp");

    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(param_dim,
                      ctx->GetInputDim("Grad"),
                      platform::errors::InvalidArgument(
                          "The shape of Intput(Param) should be equal to the "
                          "Input(Grad) of ProximalAdagrad Op. But received "
                          "Input(Param).dimensions=[%s], "
                          "Input(Grad).dimensions=[%s]",
                          param_dim,
                          ctx->GetInputDim("Grad")));

    PADDLE_ENFORCE_EQ(param_dim,
                      ctx->GetInputDim("Moment"),
                      platform::errors::InvalidArgument(
                          "The shape of Intput(Param) should be equal to the "
                          "Input(Moment) of ProximalAdagrad Op. But received "
                          "Input(Param).dimensions=[%s], "
                          "Input(Moment).dimensions=[%s]",
                          param_dim,
                          ctx->GetInputDim("Moment")));

    auto lr_dim = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(
        phi::product(lr_dim),
        1,
        platform::errors::InvalidArgument(
            "Learning Rate should be a scalar. But received dimension[%s]",
            lr_dim));

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("MomentOut", param_dim);
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Param"), ctx.GetPlace());
  }
};

class ProximalAdagradOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(Tensor, default Tensor<float>) "
             "Input parameter that has to be updated.");
    AddInput("Moment",
             "(Tensor, default Tensor<float>) "
             "Moment parameter that has to be updated.");
    AddInput("Grad",
             "(Tensor, default Tensor<float>) "
             "Input gradient of the parameter.");
    AddInput("LearningRate",
             "(Tensor, default Tensor<float>) "
             "The learning rate should be a tensor of size 1.");

    AddOutput("ParamOut", "(Tensor) Output updated parameter value.");
    AddOutput("MomentOut", "(Tensor) Output updated moment value.");

    AddAttr<float>("l1",
                   "(float, default 0.0) "
                   "L1 regularization strength.")
        .SetDefault(0.0f);
    AddAttr<float>("l2",
                   "(float, default 0.0) "
                   "L2 regularization strength.")
        .SetDefault(0.0f);
    AddComment(R"DOC(
Proximal Adagrad Optimizer.

Optimizer that implements the proximal adagrad algorithm:

$$
moment = moment + grad * grad \\
prox\_param = param - learning\_rate * grad * (1 / \sqrt{moment}) \\
param = sign(prox\_param) / (1 + learning\_rate * l2) *
        \max(|prox\_param| - learning\_rate * l1 , 0)
$$

The paper that proposed Proximal GD:
(http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf)
Here, we use the adagrad learning rate as specified here:
(http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(proximal_adagrad,
                             ops::ProximalAdagradOp,
                             ops::ProximalAdagradOpMaker);
REGISTER_OP_CPU_KERNEL(proximal_adagrad,
                       ops::ProximalAdagradOpKernel<phi::CPUContext, float>);
