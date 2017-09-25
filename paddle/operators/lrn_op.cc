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

#include "paddle/operators/lrn_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class LRNOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input(X) of LRNOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            "Output(Out) of LRNOp should not be null.");

    auto x_dims = ctx.Input<Tensor>("X")->dims();

    // ctx.Output<framework::Tensor>("Out")->Resize(
    //    {x_mat_dims[0], y_mat_dims[1]});
    // ctx.ShareLoD("X", /*->*/ "Out");
  }
};

class LRNOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LRNOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of lrn op");
    AddOutput("Out", "The output of lrn op");
  }
};

class LRNOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null");
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto out_dims = ctx.Input<Tensor>(framework::GradVarName("Out"))->dims();
    auto *x_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    if (x_grad) x_grad->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(lrn, ops::LRNOp, ops::LRNOpMaker, lrn_grad, ops::LRNOpGrad);
REGISTER_OP_CPU_KERNEL(lrn, ops::LRNKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(lrn_grad,
                       ops::LRNGradKernel<paddle::platform::CPUPlace, float>);
