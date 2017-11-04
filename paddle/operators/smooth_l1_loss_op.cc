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

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X must be initialized.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Y must be initialized.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    PADDLE_ENFORCE_EQ(x_dims, y_dims, "The shape of X and Y must be the same.");
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      "The tensor rank of X must be at least 2.");
    if (ctx->HasInput("InsideWeight")) {
      PADDLE_ENFORCE(ctx->HasInput("OutsideWeight"),
                     "If weights are provided, must specify both "
                     "inside and outside weights.");
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("InsideWeight"), x_dims,
                        "The shape of InsideWeight must be same as X.");
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("OutsideWeight"), x_dims,
                        "The shape of OutsideWeight must be same as X.");
    }

    ctx->SetOutputDim("Diff", x_dims);
    // loss is a two-rank tensor
    ctx->SetOutputDim("Out", {x_dims[0], 1});
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
             "by this tensor element by element.")
        .AsDispensable();
    AddInput("OutsideWeight",
             "Optinal input of smooth l1 loss op with the same shape as X."
             "If provided, the output smooth l1 loss will be multiplied by "
             "this tensor element by element.")
        .AsDispensable();
    AddOutput("Diff", "Intermediate variable to cache InsideWeight*(X-Y).")
        .AsIntermediate();
    AddOutput("Out", "Smooth l1 loss.");
    AddAttr<AttrType>("sigma",
                      "Hyper parameter of smooth l1 loss op."
                      "A float scalar with default value 3.0.")
        .SetDefault(3.0);
    AddComment(R"DOC(
Smooth L1 Loss Operator.

This operator computes the smooth l1 loss for input and target.
The operator takes the first dimension of input as the batch size.
For each instance, it computes the smooth l1 loss element by element first
and then sums all the losses. So the resulting output shape
is [batch_size, 1].

The equation is:
loss = $$0.5 * (\sigma * (x-y))^2$$   if $$|x - y| < 1 /({\sigma}^2)$$
       $$\frac{|x - y| - 0.5}{{\sigma}^2}$$ otherwise

)DOC");
  }
};

class SmoothL1LossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("X");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    PADDLE_ENFORCE_GE(out_dims.size(), 2,
                      "The tensor rank of Input(Out@Grad) should be 2.");
    PADDLE_ENFORCE_EQ(out_dims[0], in_dims[0],
                      "The 1st dimension of Input(Out@Grad) must be "
                      "same as input.");
    PADDLE_ENFORCE_EQ(out_dims[1], 1,
                      "The 2nd dimension of Input(Out@Grad) must be 1.");

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
