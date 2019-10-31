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

using framework::Tensor;

class SigmoidFocalLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("FgNum"), "Input(FgNum) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Label");
    auto fg_dims = ctx->GetInputDim("FgNum");

    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(rank, labels_dims.size(),
                      "Input(X) and Input(Label) shall have the same rank.");
    PADDLE_ENFORCE_EQ(fg_dims.size(), 1, "The rank of Input(FgNum) must be 1.");
    bool check = true;
    if ((!ctx->IsRuntime()) && (framework::product(x_dims) <= 0 ||
                                framework::product(labels_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                        framework::slice_ddim(labels_dims, 0, rank - 1),
                        "Input(X) and Input(Label) shall have the same shape "
                        "except the last dimension.");
    }

    PADDLE_ENFORCE_EQ(labels_dims[rank - 1], 1UL,
                      "The last dimension of input(Label) should be 1.");

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
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("FgNum"), "Input(FgNum) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Label");
    auto fg_dims = ctx->GetInputDim("FgNum");
    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(rank, labels_dims.size(),
                      "Input(X) and Input(Label) shall have the same rank.");
    PADDLE_ENFORCE_EQ(fg_dims.size(), 1, "The rank of Input(FgNum) must be 1.");
    bool check = true;
    if ((!ctx->IsRuntime()) && (framework::product(x_dims) <= 0 ||
                                framework::product(labels_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                        framework::slice_ddim(labels_dims, 0, rank - 1),
                        "Input(X) and Input(Label) shall have the same shape.");

      PADDLE_ENFORCE_EQ(labels_dims[rank - 1], 1UL,
                        "The last dimension of input(Label) should be 1.");

      PADDLE_ENFORCE_EQ(
          framework::slice_ddim(x_dims, 0, rank),
          framework::slice_ddim(dout_dims, 0, rank),
          "Input(X) and Input(Out@Grad) shall have the same shape.");
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
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("sigmoid_focal_loss_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput("FgNum", this->Input("FgNum"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sigmoid_focal_loss, ops::SigmoidFocalLossOp,
                  ops::SigmoidFocalLossOpMaker,
                  ops::SigmoidFocalLossGradOpMaker<paddle::framework::OpDesc>,
                  ops::SigmoidFocalLossGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(sigmoid_focal_loss_grad, ops::SigmoidFocalLossGradOp);
REGISTER_OP_CPU_KERNEL(
    sigmoid_focal_loss,
    ops::SigmoidFocalLossKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SigmoidFocalLossKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    sigmoid_focal_loss_grad,
    ops::SigmoidFocalLossGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SigmoidFocalLossGradKernel<paddle::platform::CPUDeviceContext,
                                    double>);
