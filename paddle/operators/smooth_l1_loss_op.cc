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
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "X must be initialized.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Y"), "Y must be initialized.");

    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    PADDLE_ENFORCE_EQ(x->dims(), y->dims(),
                      "The shape of X and Y must be the same.");
    PADDLE_ENFORCE_GE(x->dims().size(), 2,
                      "The tensor rank of X must be at least 2.");
    auto* inside_weight = ctx.Input<framework::Tensor>("InsideWeight");
    if (inside_weight) {
      auto* outside_weight = ctx.Input<framework::Tensor>("OutsideWeight");
      PADDLE_ENFORCE_NOT_NULL(outside_weight,
                              "If weights are provided, must specify both "
                              "inside and outside weights.");
      PADDLE_ENFORCE_EQ(inside_weight->dims(), x->dims(),
                        "The shape of InsideWeight must be same as X.");
      PADDLE_ENFORCE_EQ(outside_weight->dims(), x->dims(),
                        "The shape of OutsideWeight must be same as X.");
    }

    auto* diff = ctx.Output<framework::LoDTensor>("Diff");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
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
    AddInput("X",
             "The input tensor of smooth l1 loss op."
             "The rank should be greater or equal to 2 with shape "
             "[batch_size, value_dim1, value_dim2, ..., value_dimN]");
    AddInput("Y",
             "The target tensor of smooth l1 loss op "
             "with the same shape as X.");
    AddInput("InsideWeight",
             "Optional input tensor of smooth l1 loss op with the same shape "
             "as X. If provided, the result of (X - Y) will be multiplied "
             "by this tensor element by element.");
    AddInput("OutsideWeight",
             "Optinal input of smooth l1 loss op with the same shape as X."
             "If provided, the output smooth l1 loss will be multiplied by "
             "this tensor element by element.");
    AddOutput("Diff", "Intermediate variable to cache InsideWeight*(X-Y).")
        .AsIntermediate();
    AddOutput("Out", "Smooth l1 loss.");
    AddAttr<AttrType>("sigma",
                      "Hyper parameter of smooth l1 loss op."
                      "A float scalar with default value 3.0.")
        .SetDefault(3.0);
    AddComment(R"DOC(
Compute smooth l1 loss for input and target. The operator take the 1st
dimension of input as batch size. For each instance, it will compute
smooth l1 loss element by element first and sum all losses to one value.
So the output shape is [batch_size, 1].

The equation is:
loss = 0.5 * (sigma * (x-y))^2    if abs(x - y) < 1 / sigma^2
       abs(x - y) - 0.5 / sigma^2 otherwise

)DOC");
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
    auto* x_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto* y_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));

    PADDLE_ENFORCE_GE(out_dims.size(), 2,
                      "The tensor rank of Input(Out@Grad) should be 2.");
    PADDLE_ENFORCE_EQ(out_dims[0], in_dims[0],
                      "The 1st dimension of Input(Out@Grad) must be "
                      "same as input.");
    PADDLE_ENFORCE_EQ(out_dims[1], 1,
                      "The 2nd dimension of Input(Out@Grad) must be 1.");

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
