
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

#include "paddle/operators/rank_loss_op.h"

namespace paddle {
namespace operators {

class RankLossOp : public framework::OperatorWithKernel {
 public:
  RankLossOp(const std::string &type, const framework::VariableNameMap &inputs,
             const framework::VariableNameMap &outputs,
             const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    // input check
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("P"), "Input(P) shouldn't be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Oi"), "Input(Oi) shouldn't be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Oj"), "Input(Oj) shouldn't be null");
    auto p_dims = ctx.Input<framework::Tensor>("P")->dims();
    auto oi_dims = ctx.Input<framework::Tensor>("Oi")->dims();
    auto oj_dims = ctx.Input<framework::Tensor>("Oj")->dims();
    PADDLE_ENFORCE_EQ(oi_dims, oj_dims,
                      "Input(Oi) and Input(Oj) must have the same size");
    PADDLE_ENFORCE_EQ(
        p_dims, oi_dims,
        "Input(P) must have the same size with Input(Oi) & Input(Oj)");
    ctx.Output<framework::Tensor>("Out")->Resize(p_dims);
  }
};

class RankLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RankLossOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("P", "The first input of RankLoss operator.");
    AddInput("Oi", "The second input of RankLoss operator.");
    AddInput("Oj", "The third input of RankLoss operator.");
    AddOutput("Out", "The output tensor of RankLoss operator.");
    AddComment(R"DOC(RankLoss operator

A rank loss operator for learning to rank (LTR) task. This operator contains
three inputs: P, Oi, and Oj, and the rank cost can be expressed as

\f[
  C_{i,j} = -\tilde{P_{ij}} * o_{i,j} + log(1 + e^{o_{i,j}}) \\
  o_{i,j} =  o_i - o_j  \\
  \tilde{P_{i,j}} = \left \{0, 0.5, 1 \right \} \ or \ \left \{0, 1 \right \}
\f]

[1]. Chris Burges, Tal Shaked, Erin Renshaw, et al. Learning to
     Rank useing Gradient Descent.
)DOC");
  }
};

class RankLossGradOp : public framework::OperatorWithKernel {
 public:
  RankLossGradOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("P"), "Input(P) shouldn't be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Oi"), "Input(Oi) shouldn't be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Oj"), "Input(Oj) shouldn't be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) shouldn't be null.");
    auto dims = ctx.Input<framework::Tensor>("P")->dims();
    ctx.Output<framework::Tensor>(framework::GradVarName("P"))->Resize(dims);
    ctx.Output<framework::Tensor>(framework::GradVarName("Oi"))->Resize(dims);
    ctx.Output<framework::Tensor>(framework::GradVarName("Oj"))->Resize(dims);
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP(rank_loss, ops::RankLossOp, ops::RankLossOpMaker, rank_loss_grad,
            ops::RankLossGradOp);
REGISTER_OP_CPU_KERNEL(rank_loss,
                       ops::RankLossKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    rank_loss_grad, ops::RankLossGradKernel<paddle::platform::CPUPlace, float>);
