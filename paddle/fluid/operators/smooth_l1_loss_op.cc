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

#include "paddle/fluid/operators/smooth_l1_loss_op.h"
#include <memory>

namespace paddle {
namespace operators {

class SmoothL1LossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SmoothL1Loss");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "SmoothL1Loss");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    bool check = true;
    if ((!ctx->IsRuntime()) &&
        (phi::product(x_dims) <= 0 || phi::product(y_dims) <= 0)) {
      check = false;
    }
    if (check) {
      PADDLE_ENFORCE_EQ(
          x_dims, y_dims,
          platform::errors::InvalidArgument(
              "Input(X) ans Input(Y) of SmoothL1LossOp should "
              "have the same size, but received X dim is %s, Y dim is %s",
              x_dims.to_str(), y_dims.to_str()));
    }
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The tensor rank of Input(X) of SmoothL1LossOp "
                          "should not be less than 2, but received %d.",
                          x_dims.size()));
    if (ctx->HasInput("InsideWeight")) {
      PADDLE_ENFORCE_EQ(ctx->HasInput("OutsideWeight"), true,
                        platform::errors::InvalidArgument(
                            "If weights are provided, must specify both "
                            "inside and outside weights."));
      auto dims = ctx->GetInputDim("InsideWeight");
      bool check = true;
      if ((!ctx->IsRuntime()) &&
          (phi::product(dims) <= 0 || phi::product(x_dims) <= 0)) {
        check = false;
      }
      if (check) {
        PADDLE_ENFORCE_EQ(x_dims, dims,
                          platform::errors::InvalidArgument(
                              "Input(X) ans Input(InsideWeight) of "
                              "SmoothL1LossOp should have the same size, but "
                              "received X dim is %s, InsideWeight dim is %s",
                              x_dims.to_str(), dims.to_str()));
      }

      dims = ctx->GetInputDim("OutsideWeight");
      check = true;
      if ((!ctx->IsRuntime()) &&
          (phi::product(dims) <= 0 || phi::product(x_dims) <= 0)) {
        check = false;
      }
      if (check) {
        PADDLE_ENFORCE_EQ(x_dims, dims,
                          platform::errors::InvalidArgument(
                              "Input(X) ans Input(OutsideWeight) of "
                              "SmoothL1LossOp should have the same size, but "
                              "received X dim is %s, OutsideWeight dim is %s",
                              x_dims.to_str(), dims.to_str()));
      }
    }

    ctx->SetOutputDim("Diff", x_dims);
    // loss is a two-rank tensor
    ctx->SetOutputDim("Out", {x_dims[0], 1});
  }
};

class SmoothL1LossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>) A tensor with rank at least 2. "
             "The input value of smooth l1 loss op with shape "
             "[batch_size, dim1, ..., dimN].");
    AddInput("Y",
             "(Tensor, default Tensor<float>) A tensor with rank at least 2. "
             "The target value of smooth l1 loss op with same shape as X.");
    AddInput("InsideWeight",
             "(Tensor, default Tensor<float>) A tensor with rank at least 2. "
             "This input is optional and should have same shape with X. "
             "If provided, the result of (X - Y) will be multiplied "
             "by this tensor element by element.")
        .AsDispensable();
    AddInput("OutsideWeight",
             "(Tensor, default Tensor<float>) A tensor with rank at least 2. "
             "This input is optional and should have same shape with X. "
             "If provided, the out smooth l1 loss will be multiplied by this "
             "tensor element by element.")
        .AsDispensable();
    AddOutput("Diff", "Intermediate variable to cache InsideWeight * (X - Y).")
        .AsIntermediate();
    AddOutput("Out",
              "(Tensor, default Tensor<float>) A tensor with rank be 2. "
              "The output smooth l1 loss with shape [batch_size, 1].");
    AddAttr<float>("sigma",
                   "Hyper parameter of smooth l1 loss op."
                   "A float scalar with default value 3.0.")
        .SetDefault(1.0);
    AddComment(R"DOC(
Smooth L1 Loss Operator.

This operator computes the smooth l1 loss for X and Y.
The operator takes the first dimension of X and Y as batch size.
For each instance, it computes the smooth l1 loss element by element first
and then sums all the losses. So the shape of Out is [batch_size, 1].

The equation is:
$$
Out_{\sigma}(X, Y)_i = \begin{cases}
0.5 * (\sigma * (X_i - Y_i)) ^ 2
\quad |X_i - Y_i| \lt \frac{1} {{\sigma} ^ 2} \\
\frac{|X_i - Y_i| - 0.5}{{\sigma}^2},
\quad otherwise
\end{cases}
$$

In the above equation, $Out_{\sigma}(X, Y)_i$, $X_i$ and $Y_i$ represent the ith
element of Out, X and Y.

)DOC");
  }
};

class SmoothL1LossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("Diff");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    PADDLE_ENFORCE_GE(
        out_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The tensor rank of Input(Out@Grad) should be 2, but received %d.",
            out_dims.size()));
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          out_dims[0], in_dims[0],
          platform::errors::InvalidArgument(
              "The 1st dimension of Input(Out@Grad) must be "
              "same as input in SmoothL1LossGradOp, but received %d and %d.",
              out_dims[0], in_dims[0]));
      PADDLE_ENFORCE_EQ(out_dims[1], 1,
                        platform::errors::InvalidArgument(
                            "The 2nd dimension of Input(Out@Grad) must be 1 in "
                            "SmoothL1LossGradOp, but received %d.",
                            out_dims[1]));
    }

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, in_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, in_dims);
    }
  }
};

template <typename T>
class SmoothL1LossGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("smooth_l1_loss_grad");
    op->SetInput("InsideWeight", this->Input("InsideWeight"));
    op->SetInput("OutsideWeight", this->Input("OutsideWeight"));
    op->SetInput("Diff", this->Output("Diff"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(smooth_l1_loss, ops::SmoothL1LossOp, ops::SmoothL1LossOpMaker,
                  ops::SmoothL1LossGradMaker<paddle::framework::OpDesc>,
                  ops::SmoothL1LossGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(smooth_l1_loss_grad, ops::SmoothL1LossGradOp);
REGISTER_OP_CPU_KERNEL(
    smooth_l1_loss,
    ops::SmoothL1LossKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    smooth_l1_loss_grad,
    ops::SmoothL1LossGradKernel<paddle::platform::CPUDeviceContext, float>);
