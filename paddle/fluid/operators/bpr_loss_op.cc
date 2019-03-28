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

#include "paddle/fluid/operators/bpr_loss_op.h"

namespace paddle {
namespace operators {

class BprLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(rank, label_dims.size(),
                      "Input(X) and Input(Label) shall have the same rank.");
    PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                      framework::slice_ddim(label_dims, 0, rank - 1),
                      "Input(X) and Input(Label) shall have the same shape "
                      "except the last dimension.");

    auto y_dims = x_dims;
    y_dims[rank - 1] = 1;
    ctx->SetOutputDim("Y", y_dims);
    ctx->ShareLoD("X", /*->*/ "Y");
  }

 protected:
  // Explicitly set that the data type of computation kernel of Seq-bpr
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   platform::CPUPlace());
  }
};

class BprLossGradientOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@GRAD) shoudl be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    auto dy_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(dy_dims.size(), rank,
                      "Input(Y@Grad) and Input(X) should have the same rank.");
    PADDLE_ENFORCE_EQ(label_dims.size(), rank,
                      "Input(Label) and Input(X) should have the same rank.");
    PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                      framework::slice_ddim(label_dims, 0, rank - 1),
                      "The Input(X) and Input(Label) should have the same "
                      "shape except the last dimension.");
    PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                      framework::slice_ddim(dy_dims, 0, rank - 1),
                      "The Input(X) and Input(Y@Grad) should have the same "
                      "shape except the last dimension.");
    PADDLE_ENFORCE_EQ(dy_dims[rank - 1], 1,
                      "The last dimension of Input(Y@Grad) should be 1.");
    PADDLE_ENFORCE_EQ(label_dims[rank - 1], 1,
                      " the last dimension of Input(Label) should be 1.");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

 protected:
  // Explicitly set that the data type of computation kernel of cross_entropy
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   platform::CPUPlace());
  }
};

class BprLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), a tensor whose last dimension "
             "size is equal to the number of classes. This input is a "
             "real number.");
    AddInput(
        "Label",
        "(Tensor), the tensor which represents the ground truth. It has the "
        "same shape with 'X' except the last dimension. the last dimension "
        "size is 1.");
    AddOutput("Y",
              "(Tensor, default Tensor<float>), a tensor whose shape is same "
              "with 'X' except that the last dimension size is 1. It "
              "represents the sequence bpr loss.");
    AddComment(R"DOC(
Bayesian Personalized Ranking Loss Operator.

This operator belongs to pairwise ranking loss. Label is the desired item.
The loss at a given point in one session is defined as:
$Y[i] = -\frac{1}{N_{i}} * \sum_{j=0}^{N_{i}}\log(\sigma(X[i, Label[i]]-X[i, j]))$

Learn more details by reading paper <session-based recommendations with recurrent
neural networks>(https://arxiv.org/abs/1511.06939)

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPUCtx = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(bpr_loss, ops::BprLossOp, ops::BprLossOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(bpr_loss_grad, ops::BprLossGradientOp);
REGISTER_OP_CPU_KERNEL(bpr_loss, ops::BprLossOpKernel<CPUCtx, float>,
                       ops::BprLossOpKernel<CPUCtx, double>);
REGISTER_OP_CPU_KERNEL(bpr_loss_grad,
                       ops::BprLossGradientOpKernel<CPUCtx, float>,
                       ops::BprLossGradientOpKernel<CPUCtx, double>);
