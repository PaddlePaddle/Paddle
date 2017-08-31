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

#include "paddle/operators/element_wise_mul_op.h"

namespace paddle {
namespace operators {

class ElemWiseMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto dim0 = ctx.Input<Tensor>("X")->dims();
    auto dim1 = ctx.Input<Tensor>("Y")->dims();
    PADDLE_ENFORCE_EQ(
        dim0, dim1,
        "First matrix's dims must be equal with second matrix's dims.");
    ctx.Output<Tensor>("Out")->Resize(dim0);
  }
};

class ElemWiseMulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ElemWiseMulOpMaker(framework::OpProto *proto,
                     framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of element-wise mul op");
    AddInput("Y", "The second input of element-wise mul op");
    AddOutput("Out", "The output of element-wise mul op");
    AddComment(R"DOC(
Element-wise mul operator.

The equation is: Out = X ⊙ Y
)DOC");
  }
};

class ElemWiseMulOpGrad : public framework::OperatorWithKernel {
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
    auto *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *y_grad = ctx.Output<Tensor>(framework::GradVarName("Y"));
    PADDLE_ENFORCE(x_dims == out_dims, "Out@GRAD must equal to X dims");
    PADDLE_ENFORCE(y_dims == out_dims, "Out@GRAD must equal to Y dims");

    x_grad->Resize(x_dims);
    y_grad->Resize(y_dims);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(elemwisemul, ops::ElemWiseMulOp, ops::ElemWiseMulOpMaker,
            elemwisemul_grad, ops::ElemWiseMulOpGrad);
REGISTER_OP_CPU_KERNEL(
    elemwisemul, ops::ElemWiseMulKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    elemwisemul_grad,
    ops::ElemWiseMulGradKernel<paddle::platform::CPUPlace, float>);
