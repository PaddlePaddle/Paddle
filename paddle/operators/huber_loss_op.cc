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

#include "paddle/operators/huber_loss_op.h"

namespace paddle {
namespace operators {

class HuberLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) must be initialized.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Y"), "Input(Y) must be initialized.");

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");

    PADDLE_ENFORCE_EQ(x->dims(), y->dims());
    PADDLE_ENFORCE_EQ(framework::arity(x->dims()), 2,
                      "The rank of Input(X) must be 2 and the shape is "
                      "[batch_size, 1].");
    PADDLE_ENFORCE_EQ(x->dims()[1], 1,
                      "Each row of Input(X) contains a real value, "
                      "so the 2nd dimension of Input(X) must be 1.");

    ctx.Output<Tensor>("Residual")->Resize(x->dims());
    ctx.Output<Tensor>("Out")->Resize({x->dims()[0], 1});
  }
};

template <typename AttrType>
class HuberLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  HuberLossOpMaker(framework::OpProto* proto,
                   framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "The input value of huber loss op."
             "X is a 2-D tensor with shape [batch_size, 1].");
    AddInput("Y",
             "The target value of huber loss op."
             "Y is a 2-D tensor with shape [batch_size, 1].");
    AddOutput("Residual",
              "Intermediate tensor to cache residual value of Y and X."
              "The shape is same as Input(X) and will be reused in backward.")
        .AsIntermediate();
    AddOutput("Out",
              "The output tensor with shape [batch_size, 1] which represents "
              "the huber loss.");
    AddAttr<AttrType>("delta", "Hyper parameter in huber loss.");
    AddComment(R"DOC(
Huber loss is a loss function used in robust regression. We define X as the
input value and Y as the target value. Huber loss can evaluate the fitness of
X to Y. Different from MSE loss, Huber loss is more robust for outliers. The
shape of X and Y are [batch_size, 1]. The equation is:

L_{\delta}(y, f(x)) =
\begin{cases}
0.5 * (y - f(x))^2, \quad |y - f(x)| \leq \delta \\
\delta * (|y - f(x)| - 0.5 * \delta),   \quad otherwise
\end{cases}

)DOC");
  }
};

class HuberLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* residual = ctx.Input<Tensor>("Residual");
    auto* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* y_grad = ctx.Output<Tensor>(framework::GradVarName("Y"));

    PADDLE_ENFORCE_NOT_NULL(x, "Input(X) should not be null.");
    PADDLE_ENFORCE_NOT_NULL(y, "Input(Y) should not be null.");
    PADDLE_ENFORCE_NOT_NULL(residual, "Input(Residual) should not be null.");
    PADDLE_ENFORCE_NOT_NULL(out_grad, "Input(Out@GRAD) should not be null.");

    PADDLE_ENFORCE_EQ(residual->dims(), x->dims());
    PADDLE_ENFORCE_EQ(out_grad->dims(), x->dims());

    if (x_grad) x_grad->Resize(x->dims());
    if (y_grad) y_grad->Resize(y->dims());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(huber_loss, ops::HuberLossOp, ops::HuberLossOpMaker<float>,
            huber_loss_grad, ops::HuberLossGradOp);
REGISTER_OP_CPU_KERNEL(huber_loss,
                       ops::HuberLossKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    huber_loss_grad,
    ops::HuberLossGradKernel<paddle::platform::CPUPlace, float>);
