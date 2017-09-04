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

#include "paddle/operators/row_l2_norm_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class RowL2NormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null");
    auto in_dims = ctx.Input<Tensor>("X")->dims();
    PADDLE_ENFORCE_EQ(in_dims.size(), 2,
                      "input X(%s) should be a tensor with 2 dims, a matrix",
                      ctx.op().Input("X"));
    ctx.Output<Tensor>("L2_Norm")->Resize({in_dims[0], 1});
    ctx.Output<Tensor>("Out")->Resize(in_dims);
  }
};

class RowL2NormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RowL2NormOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of RowL2NormOp");
    AddOutput("L2_Norm", "Buffering l2-norm of each row").AsIntermediate();
    AddOutput("Out", "The output of RowL2NormOp");
    AddComment(R"DOC(
Given a matrix, apply L2-normalization on the row.

The equation is: out[i] = \frac{in[i]}{\sqrt{\sum_{k=1}^N in[k]^{2}}}
)DOC");
  }
};

class RowL2NormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null");
    auto in_dims = ctx.Input<Tensor>("X")->dims();
    auto out_dims = ctx.Input<Tensor>(framework::GradVarName("Out"))->dims();
    PADDLE_ENFORCE(in_dims[0] == out_dims[0] && in_dims[1] == out_dims[1],
                   "Out@GRAD dims must equal to X dims");
    auto *in_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    if (in_grad != nullptr) in_grad->Resize(in_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(row_l2_norm, ops::RowL2NormOp, ops::RowL2NormOpMaker,
            ops::RowL2NormGradOp);
REGISTER_OP_CPU_KERNEL(row_l2_norm,
                       ops::RowL2NormKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    row_l2_norm_grad,
    ops::RowL2NormGradKernel<paddle::platform::CPUPlace, float>);
