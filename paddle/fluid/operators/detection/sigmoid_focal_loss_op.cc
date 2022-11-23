/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection/sigmoid_focal_loss_op.h"

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class SigmoidFocalLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "sigmoid_focal_loss");
    OP_INOUT_CHECK(
        ctx->HasInput("Label"), "Input", "Label", "sigmoid_focal_loss");
    OP_INOUT_CHECK(
        ctx->HasInput("FgNum"), "Input", "FgNum", "sigmoid_focal_loss");
    OP_INOUT_CHECK(
        ctx->HasOutput("Out"), "Output", "Out", "sigmoid_focal_loss");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Label");
    auto fg_dims = ctx->GetInputDim("FgNum");

    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(
        rank,
        labels_dims.size(),
        platform::errors::InvalidArgument(
            "The rank of Input(X) should be equal to the rank of Input(Label), "
            "but received X rank is:%d, X shape is:[%s], "
            "Label rank is:%d, Label shape is:[%s].",
            rank,
            x_dims,
            labels_dims.size(),
            labels_dims));
    PADDLE_ENFORCE_EQ(
        fg_dims.size(),
        1,
        platform::errors::InvalidArgument(
            "The rank of Input(FgNum) must be 1, but received FgNum rank is "
            ":%d, FgNum shape is:[%s].",
            fg_dims.size(),
            fg_dims));
    bool check = true;
    if ((!ctx->IsRuntime()) &&
        (phi::product(x_dims) <= 0 || phi::product(labels_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(
          phi::slice_ddim(x_dims, 0, rank - 1),
          phi::slice_ddim(labels_dims, 0, rank - 1),
          platform::errors::InvalidArgument(
              "Input(X) and Input(Label) should have the same shape "
              "except the last dimension, but received X shape is:[%s], "
              "Label shape is:[%s].",
              x_dims,
              labels_dims));
    }

    PADDLE_ENFORCE_EQ(
        labels_dims[rank - 1],
        1UL,
        platform::errors::InvalidArgument(
            "The last dimension of Input(Label) should be 1, but received "
            "Label shape is:[%s].",
            labels_dims));

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class SigmoidFocalLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "sigmoid_focal_loss");
    OP_INOUT_CHECK(
        ctx->HasInput("Label"), "Input", "Label", "sigmoid_focal_loss");
    OP_INOUT_CHECK(
        ctx->HasInput("FgNum"), "Input", "FgNum", "sigmoid_focal_loss");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "sigmoid_focal_loss");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   "X@GRAD",
                   "sigmoid_focal_loss");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Label");
    auto fg_dims = ctx->GetInputDim("FgNum");
    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(
        rank,
        labels_dims.size(),
        platform::errors::InvalidArgument(
            "The rank of Input(X) should be equal to the rank of Input(Label), "
            "but received X rank is:%d, X shape is:[%s], "
            "Label rank is:%d, Label shape is:[%s].",
            rank,
            x_dims,
            labels_dims.size(),
            labels_dims));
    PADDLE_ENFORCE_EQ(
        fg_dims.size(),
        1,
        platform::errors::InvalidArgument(
            "The rank of Input(FgNum) must be 1, but received FgNum rank is "
            ":%d, FgNum shape is:[%s].",
            fg_dims.size(),
            fg_dims));
    bool check = true;
    if ((!ctx->IsRuntime()) &&
        (phi::product(x_dims) <= 0 || phi::product(labels_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(
          phi::slice_ddim(x_dims, 0, rank - 1),
          phi::slice_ddim(labels_dims, 0, rank - 1),
          platform::errors::InvalidArgument(
              "Input(X) and Input(Label) should have the same shape "
              "except the last dimension, but received X shape is:[%s], "
              "Label shape is:[%s].",
              x_dims,
              labels_dims));

      PADDLE_ENFORCE_EQ(
          labels_dims[rank - 1],
          1UL,
          platform::errors::InvalidArgument(
              "The last dimension of Input(Label) should be 1, but received "
              "Label shape is:[%s].",
              labels_dims));

      PADDLE_ENFORCE_EQ(phi::slice_ddim(x_dims, 0, rank),
                        phi::slice_ddim(dout_dims, 0, rank),
                        platform::errors::InvalidArgument(
                            "Input(X) and Input(Out@Grad) should have the same "
                            "shape, but received "
                            "X shape is:[%s], Out@Grad shape is:[%s].",
                            x_dims,
                            dout_dims));
    }

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class SigmoidFocalLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), a 2-D tensor with shape [N, D], "
             "where N is the batch size and D is the number of classes "
             "(excluding background). This input is a tensor of logits "
             "computed by the previous operator.");
    AddInput("Label",
             "(Tensor, default Tensor<int>), a 2-D tensor with shape [N, 1]. "
             "This input is a tensor of probabilistic labels.");
    AddInput("FgNum",
             "(Tensor, default Tensor<int>), a 1-D tensor with shape [1]. "
             "This input is the number of foreground.");
    AddOutput(
        "Out",
        "(Tensor, default Tensor<float>), a 2-D tensor with shape [N, D]. "
        "This output is the focal loss.");
    AddAttr<float>(
        "gamma",
        "Hyper-parameter of sigmoid focal loss op, which is to balance the "
        "easy and hard examples. "
        "A float scalar with default value 2.0.")
        .SetDefault(2.0);
    AddAttr<float>(
        "alpha",
        "Hyper-parameter of sigmoid focal loss op, which is to balance the "
        "positive and negative examples. "
        "A float scalar with default value 0.5.")
        .SetDefault(0.25);
    AddComment(R"DOC(
Sigmoid Focal Loss Operator.

Focal loss is used to address the foreground-background class imbalance existed
on the training phase of one-stage detectors. This operator computes the sigmoid
value for each element in the input tensor, after which focal loss is measured.

The focal loss is given as follows:

$$Loss_j = (-Label_j * alpha * \pow(1 - \sigma(X_j), gamma) * \log(\sigma(X_j)) -
(1 - Labels_j) * (1 - alpha) * \pow(\sigma(X_j), gamma) * \log(1 - \sigma(X_j)))
/ FgNum, j = 1,...,K$$

We know that $$\sigma(X_j) = \\frac{1}{1 + \exp(-X_j)}$$.

)DOC");
  }
};

template <typename T>
class SigmoidFocalLossGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sigmoid_focal_loss_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput("FgNum", this->Input("FgNum"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sigmoid_focal_loss,
                  ops::SigmoidFocalLossOp,
                  ops::SigmoidFocalLossOpMaker,
                  ops::SigmoidFocalLossGradOpMaker<paddle::framework::OpDesc>,
                  ops::SigmoidFocalLossGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(sigmoid_focal_loss_grad, ops::SigmoidFocalLossGradOp);
REGISTER_OP_CPU_KERNEL(sigmoid_focal_loss,
                       ops::SigmoidFocalLossKernel<phi::CPUContext, float>,
                       ops::SigmoidFocalLossKernel<phi::CPUContext, double>);
REGISTER_OP_CPU_KERNEL(
    sigmoid_focal_loss_grad,
    ops::SigmoidFocalLossGradKernel<phi::CPUContext, float>,
    ops::SigmoidFocalLossGradKernel<phi::CPUContext, double>);
