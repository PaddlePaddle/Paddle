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

#include "paddle/operators/margin_rank_loss_op.h"

namespace paddle {
namespace operators {

class MarginRankLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    // input check
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Label"),
                            "Input(Label) shouldn't be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X1"), "Input(X1) shouldn't be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X2"), "Input(X2) shouldn't be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            "Output(X2) shouldn't be null.");
    auto label_dims = ctx.Input<framework::Tensor>("Label")->dims();
    auto x1_dims = ctx.Input<framework::Tensor>("X1")->dims();
    auto x2_dims = ctx.Input<framework::Tensor>("X2")->dims();
    PADDLE_ENFORCE((label_dims == x1_dims) && (x1_dims == x2_dims) &&
                       (label_dims.size() == 2) && (label_dims[1] == 1),
                   "All inputs must be vector with the same size.");
    auto act_t = ctx.Output<framework::LoDTensor>("Activated");
    auto out_t = ctx.Output<framework::LoDTensor>("Out");
    if (act_t) {
      act_t->Resize(label_dims);
    }
    if (out_t) {
      out_t->Resize(label_dims);
    }
  }
};

template <typename T>
class MarginRankLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MarginRankLossOpMaker(framework::OpProto *proto,
                        framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X1",
             "(2-D tensor with shape [batch_size x 1]) In pairwise ranking, "
             "X1 is the score for one item to be ranked.");
    AddInput("X2",
             "(2-D tensor with shape [batch_size x 1]) In pairwise ranking, "
             "X2 is the score for another item to be ranked.");
    AddInput("Label",
             "(2-D tensor with shape [batch_size x 1]) "
             "The label indicating X1 ranked higher than X2 or not, "
             "can only be +1 or -1.");
    AddAttr<T>("margin", "(scalar, default 0) Margin for MarginRankLossOp.")
        .SetDefault(static_cast<T>(0));
    AddOutput("Activated",
              "(2-D tensor with shape [batch_size x 1]) Intermediate tensor "
              "to indicate whether each element of Output(Out) is activated.")
        .AsIntermediate();
    AddOutput("Out",
              "(2-D tensor with shape [batch_size x 1])"
              "The output loss of MarginRankLoss operator");
    AddComment(R"DOC(

MarginRankLoss operator measures the loss given a pair of input {`X1`, `X2`}
and the `Label` with attribute `margin`, where `Label = +1` indicating X1 is
ranked higher than `X2`, otherwise `Label = -1`. The loss turns out

loss(X1, X2, Label) = max(0, -Label * (X1 - X2) + margin)

The attribute `margin` involved here helps make the predictions more robust.
Only when the difference between `X1` and `X2` is greater than `margin`, it is
possible for these two items contribute to the final loss.

For batch input with size `batch_size`, `X1`, `X2` and `Label`
all have the same shape [batch_size x 1].

)DOC");
  }
};

class MarginRankLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Label"),
                            "Input(Label) shouldn't be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X1"), "Input(X1) shouldn't be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X2"), "Input(X2) shouldn't be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) shouldn't be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Activated"),
                            "Intermediate(Activated) shouldn't be null.");
    auto dims = ctx.Input<framework::Tensor>("X1")->dims();
    auto *x1_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X1"));
    auto *x2_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X2"));
    if (x1_grad) {
      x1_grad->Resize(dims);
    }
    if (x2_grad) {
      x2_grad->Resize(dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP(margin_rank_loss, ops::MarginRankLossOp,
            ops::MarginRankLossOpMaker<float>, margin_rank_loss_grad,
            ops::MarginRankLossGradOp);
REGISTER_OP_CPU_KERNEL(
    margin_rank_loss,
    ops::MarginRankLossKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    margin_rank_loss_grad,
    ops::MarginRankLossGradKernel<paddle::platform::CPUPlace, float>);
