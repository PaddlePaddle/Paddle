/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/bce_loss_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class BCELossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BCELoss");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "BCELoss");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "BCELoss");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Label");

    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(rank, labels_dims.size(),
                      platform::errors::InvalidArgument(
                          "Input(X) and Input(Label) shall have the same rank."
                          "But received: the rank of Input(X) is [%d], "
                          "the rank of Input(Label) is [%d].",
                          rank, labels_dims.size()));

    bool check = true;
    if ((!ctx->IsRuntime()) && (framework::product(x_dims) <= 0 ||
                                framework::product(labels_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(x_dims, labels_dims,
                        platform::errors::InvalidArgument(
                            "Input(X) and Input(Label) shall have the same "
                            "shape. But received: the shape of Input(X) is "
                            "[%s], the shape of Input(Label) is [%s].",
                            x_dims, labels_dims));
    }

    ctx->ShareDim("X", "Out");
    ctx->ShareLoD("X", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class BCELossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BCELossGrad");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "BCELossGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "BCELossGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   framework::GradVarName("X"), "BCELossGrad");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Label");
    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    bool check = true;
    if ((!ctx->IsRuntime()) && (framework::product(x_dims) <= 0 ||
                                framework::product(labels_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(x_dims, labels_dims,
                        platform::errors::InvalidArgument(
                            "Input(X) and Input(Label) shall have the same "
                            "shape. But received: the shape of Input(X) is "
                            "[%s], the shape of Input(Label) is [%s].",
                            x_dims, labels_dims));

      PADDLE_ENFORCE_EQ(x_dims, dout_dims,
                        platform::errors::InvalidArgument(
                            "Input(X) and Input(Out@Grad) shall have the same "
                            "shape. But received: the shape of Input(X) is "
                            "[%s], the shape of Input(Out@Grad) is [%s].",
                            x_dims, dout_dims));
    }

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class BCELossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), the input is a tensor of logits"
             "computed by the previous operator, which is always the result of"
             "a sigmoid operator. Input must between in 0 and 1.");
    AddInput("Label",
             "(Tensor, default Tensor<float>), have same shape with input"
             "label should between in 0 and 1.");
    AddOutput("Out",
              "(Tensor, default Tensor<float>), have same shape with"
              "input");
    AddComment(R"DOC(
BinaryCrossEntropy operator.

This measures the element-wise probability error in classification tasks
in which each class is independent.

The logitstic loss is given as follows:
      $$loss = -Label * \log(X) - (1 - Label) * \log(1 - X)$$
)DOC");
  }
};

template <typename T>
class BCELossGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("bce_loss_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

DECLARE_INPLACE_OP_INFERER(BCELossInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(BCELossGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(bce_loss, ops::BCELossOp, ops::BCELossOpMaker,
                  ops::BCELossGradOpMaker<paddle::framework::OpDesc>,
                  ops::BCELossGradOpMaker<paddle::imperative::OpBase>,
                  ops::BCELossInplaceInferer);
REGISTER_OPERATOR(bce_loss_grad, ops::BCELossGradOp,
                  ops::BCELossGradInplaceInferer);
REGISTER_OP_CPU_KERNEL(
    bce_loss, ops::BCELossOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BCELossOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    bce_loss_grad,
    ops::BCELossGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BCELossGradOpKernel<paddle::platform::CPUDeviceContext, double>);
