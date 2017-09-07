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

#include "paddle/operators/smooth_l1_loss_op.h"

namespace paddle {
namespace operators {

class SmoothL1LossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input of SmoothL1LossOp must be initialized.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Y"),
                            "Target of SmoothL1LossOp must be initialized.");

    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    PADDLE_ENFORCE_EQ(x->dims(), y->dims(),
                      "Dimensions of SmoothL1LossOp's input and target "
                      "must be same.");
    PADDLE_ENFORCE_GE(framework::arity(x->dims()), 2,
                      "Tensor rank of SmoothL1LossOp's input must be "
                      "at least 2.");
    auto* inside_weight = ctx.Input<framework::Tensor>("InsideWeight");
    if (inside_weight) {
      auto* outside_weight = ctx.Input<framework::Tensor>("OutsideWeight");
      PADDLE_ENFORCE_NOT_NULL(outside_weight,
                              "If weights are provided, must specify both "
                              "inside and outside weights.");
      PADDLE_ENFORCE_EQ(inside_weight->dims(), x->dims(),
                        "Dimensions of inside weight must be same with input.");
      PADDLE_ENFORCE_EQ(
          outside_weight->dims(), x->dims(),
          "Dimensions of outside weight must be same with input.");
    }

    auto* diff = ctx.Output<framework::Tensor>("diff");
    auto* out = ctx.Output<framework::Tensor>("Out");
    diff->Resize(x->dims());
    // loss is a two-rank tensor
    out->Resize({x->dims()[0], 1});
  }
};

template <typename AttrType>
class SmoothL1LossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SmoothL1LossOpMaker(framework::OpProto* proto,
                      framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of SmoothL1LossOp.");
    AddInput("Y", "Target of SmoothL1LossOp.");
    AddInput("InsideWeight", "Optional input to scale (X-Y).");
    AddInput("OutsideWeight", "Optinal input to scale smooth l1 loss.");
    AddOutput("diff", "Intermediate variable to cache Win*(X-Y).")
        .AsIntermediate();
    AddOutput("Out", "Final smooth l1 loss of inputs.");
    AddComment(R"DOC(
Compute SmoothL1Loss for input and target.

The equation is: Out = 0.5 * (sigma * (X - Y)) ^ 2  if abs(X - Y) < 1 / sigma^2
                       abs(X - Y) - 0.5 / sigma^2   otherwise
)DOC");
    AddAttr<AttrType>("sigma", "Hyper parameter, default value is 3.0 .")
        .SetDefault(3.0);
  }
};

class SmoothL1LossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {
    auto in_dims = ctx.Input<framework::Tensor>("X")->dims();
    auto out_dims =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"))->dims();
    auto* x_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* y_grad = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));

    PADDLE_ENFORCE_GE(framework::arity(out_dims), 2,
                      "Tensor rank of output gradient should be 2.");
    PADDLE_ENFORCE_EQ(out_dims[0], in_dims[0],
                      "First dimension of ouptut gradient must be "
                      "same with input.");
    PADDLE_ENFORCE_EQ(out_dims[1], 1,
                      "Second dimension of output gradient must be 1.");

    if (x_grad) x_grad->Resize(in_dims);
    if (y_grad) y_grad->Resize(in_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(smooth_l1_loss, ops::SmoothL1LossOp,
            ops::SmoothL1LossOpMaker<float>, smooth_l1_loss_grad,
            ops::SmoothL1LossGradOp);
REGISTER_OP_CPU_KERNEL(
    smooth_l1_loss, ops::SmoothL1LossKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    smooth_l1_loss_grad,
    ops::SmoothL1LossGradKernel<paddle::platform::CPUPlace, float>);
