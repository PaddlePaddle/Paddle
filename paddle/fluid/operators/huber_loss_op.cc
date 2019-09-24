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

#include "paddle/fluid/operators/huber_loss_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class HuberLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input(X) must be initialized.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Y"), true,
                      "Input(Y) must be initialized.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    int rank = x_dims.size();

    if (rank == y_dims.size()) {
      PADDLE_ENFORCE_EQ(y_dims[rank - 1], 1U,
                        "The last dimension of Input(Y) should be equal to 1.");
    } else {
      PADDLE_ENFORCE_EQ(rank, y_dims.size() + 1,
                        "The rank of Input(X) should be equal to "
                        "the rank of Input(Y) plus 1.");
    }
    bool contain_unknown_dim = framework::contain_unknown_dim(x_dims) ||
                               framework::contain_unknown_dim(y_dims);
    if (ctx->IsRuntime() || !contain_unknown_dim) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                        framework::slice_ddim(y_dims, 0, rank - 1),
                        "The Input(X) and Input(Label) should have the same "
                        "shape except the last dimension.");
    }

    auto out_dims = y_dims;
    ctx->SetOutputDim("Residual", out_dims);
    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", "Out");
  }
};

template <typename AttrType>
class HuberLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input value of huber loss op."
             "X is a 2-D tensor with shape [batch_size, 1].");
    AddInput("Y",
             "The target value of huber loss op."
             "Y is a 2-D tensor with shape [batch_size, 1].");
    AddOutput("Residual",
              "Intermediate tensor to cache residual value between Y and X."
              "The shape is same as Input(X) and will be reused in backward.")
        .AsIntermediate();
    AddOutput("Out",
              "The output tensor with shape [batch_size, 1] "
              "which represents the huber loss.");
    AddAttr<AttrType>("delta", "Hyper parameter in huber loss.");
    AddComment(R"DOC(
HuberLoss Operator.

Huber loss is a loss function used in robust regression. We define X as the
input value and Y as the target value. Huber loss can evaluate the fitness of
X to Y. Different from MSE loss, Huber loss is more robust for outliers. The
shape of X and Y are [batch_size, 1]. The equation is:

$$
Out_{\delta}(X, Y)_i =
\begin{cases}
0.5 * (Y_i - X_i)^2,
\quad |Y_i - X_i| \leq \delta \\
\delta * (|Y_i - X_i| - 0.5 * \delta),
\quad otherwise
\end{cases}
$$

In the above equation, $Out_\delta(X, Y)_i$, $X_i$ and $Y_i$ represent the ith
element of Out, X and Y.

)DOC");
  }
};

class HuberLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      "Input(Out@GRAD) should not be null.");

    auto residual_dims = ctx->GetInputDim("Residual");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, residual_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, residual_dims);
    }
  }
};

class HuberLossGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("huber_loss_grad");
    op->SetInput("Residual", Output("Residual"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(huber_loss, ops::HuberLossOp, ops::HuberLossOpMaker<float>,
                  ops::HuberLossGradOpDescMaker);
REGISTER_OPERATOR(huber_loss_grad, ops::HuberLossGradOp);
REGISTER_OP_CPU_KERNEL(
    huber_loss, ops::HuberLossKernel<paddle::platform::CPUDeviceContext, float>,
    ops::HuberLossKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    huber_loss_grad,
    ops::HuberLossGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::HuberLossGradKernel<paddle::platform::CPUDeviceContext, double>);
