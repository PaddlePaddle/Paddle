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

#include "paddle/operators/elementwise_mul_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class ElementWiseMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input(X) of ElementWiseMulOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Y"),
                            "Input(Y) of ElementWiseMulOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(
        ctx.OutputVar("Out"),
        "Output(Out) of ElementWiseMulOp should not be null.");

    auto x_dim = ctx.Input<Tensor>("X")->dims();
    auto y_dim = ctx.Input<Tensor>("Y")->dims();
    PADDLE_ENFORCE_GE(x_dim.size(), y_dim.size(),
                      "Rank of first input must >= rank of second input.")
    ctx.Output<framework::LoDTensor>("Out")->Resize(x_dim);
  }
};

class ElementWiseMulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ElementWiseMulOpMaker(framework::OpProto *proto,
                        framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of elementwise mul op");
    AddInput("Y", "The second input of elementwise mul op");
    AddAttr<int>("axis",
                 R"DOC(
When shape(Y) does not equal shape(X),Y will be broadcasted 
to match the shape of X and axis should be dimension index Y in X
        )DOC")
        .SetDefault(-1)
        .EqualGreaterThan(-1);

    AddOutput("Out", "The output of elementwise mul op");
    AddComment(R"DOC(
Limited elementwise multiple operator.The equation is: Out = X ⊙ Y.
1. The shape of Y should be same with X or
2. Y's shape is a subset of X. 
   Y will be broadcasted to match the shape of X and axis should be dimension index Y in X.
   example:
      shape(X) = (2, 3, 4, 5), shape(Y) = (,)
      shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
      shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5)
      shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
      shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
)DOC");
  }
};

class ElementWiseMulOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null");

    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto y_dims = ctx.Input<Tensor>("Y")->dims();
    auto out_dims = ctx.Input<Tensor>(framework::GradVarName("Out"))->dims();
    auto *x_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto *y_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));

    PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                      "Rank of first input must >= rank of second input.")

    if (x_grad) {
      x_grad->Resize(x_dims);
    }

    if (y_grad) {
      y_grad->Resize(y_dims);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(elementwise_mul, ops::ElementWiseMulOp, ops::ElementWiseMulOpMaker,
            elementwise_mul_grad, ops::ElementWiseMulOpGrad);
REGISTER_OP_CPU_KERNEL(
    elementwise_mul,
    ops::ElementWiseMulKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    elementwise_mul_grad,
    ops::ElementWiseMulGradKernel<paddle::platform::CPUPlace, float>);
