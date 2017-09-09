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
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "X must be initialized.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Y"), "Y must be initialized.");

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");

    PADDLE_ENFORCE_EQ(x->dims(), y->dims(),
                      "Dimensions of X and Y must be the same.");
    // we constraint shape of X to (N, 1), may expand to (N, x, ...) if needed
    PADDLE_ENFORCE_EQ(framework::arity(x->dims()), 2,
                      "Tensor rank of X must be 2.");
    PADDLE_ENFORCE_EQ(x->dims()[1], 1, "Second dimension of X must be 1.");

    ctx.Output<Tensor>("residual")->Resize(x->dims());
    ctx.Output<Tensor>("Out")->Resize({x->dims()[0], 1});
  }
};

template <typename AttrType>
class HuberLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  HuberLossOpMaker(framework::OpProto* proto,
                   framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input value of HuberLossOp.");
    AddInput("Y", "Target value of HuberLossOp.");
    AddOutput("residual",
              "Save residual value between Y and X. "
              "Will be reused in backward.")
        .AsIntermediate();
    AddOutput("Out", "Huber loss between input and target.");
    AddAttr<AttrType>("delta", "Hyper parameter in huber loss.");
    AddComment(R"DOC(
Huber loss is a loss function used in robust regression. We constrain shape of
input to (N, 1). The formulation is:

L_delta(y, f(x)) = 0.5 * (y - f(x))^2                  for |y - f(x)| <= delta,
                   delta * (|y - f(x)| - 0.5 * delta)  otherwise.

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
    auto* residual = ctx.Input<Tensor>("residual");
    auto* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* y_grad = ctx.Output<Tensor>(framework::GradVarName("Y"));

    PADDLE_ENFORCE_NOT_NULL(x, "Input X must not be null.");
    PADDLE_ENFORCE_NOT_NULL(y, "Target Y must not be null.");
    PADDLE_ENFORCE_NOT_NULL(residual, "Residual value must not be null.");
    PADDLE_ENFORCE_NOT_NULL(out_grad, "Out gradient must not be null.");

    PADDLE_ENFORCE_EQ(residual->dims(), x->dims(),
                      "Dimension of X and residual value must be the same.");
    PADDLE_ENFORCE_EQ(
        out_grad->dims(), x->dims(),
        "Dimension of Out gradient and X must be the same (N*1).");

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
