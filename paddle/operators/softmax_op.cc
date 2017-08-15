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

#include "paddle/operators/softmax_op.h"

namespace paddle {
namespace operators {

class SoftmaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(ctx.Input<Tensor>("X")->dims().size() == 2UL,
                   "The input of softmax op must be matrix");
    ctx.Output<Tensor>("Y")->Resize(ctx.Input<Tensor>("X")->dims());
  }
};

class SoftmaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SoftmaxOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "input of softmax");
    AddOutput("Y", "output of softmax");
    AddComment("Softmax Op");
  }
};

class SoftmaxOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(ctx.InputVar("Y") != nullptr, "Input(Y) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Y")),
                            "Input(Y@GRAD) should not be null");
    PADDLE_ENFORCE(ctx.Input<Tensor>("Y")->dims() ==
                       ctx.Input<Tensor>(framework::GradVarName("Y"))->dims(),
                   "the shape of Input(0) and Input(1) should be the same");
    ctx.Output<Tensor>(framework::GradVarName("X"))
        ->Resize(ctx.Input<Tensor>("Y")->dims());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(softmax, ops::SoftmaxOp, ops::SoftmaxOpMaker, softmax_grad,
            ops::SoftmaxOpGrad);
REGISTER_OP_CPU_KERNEL(softmax,
                       ops::SoftmaxKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    softmax_grad, ops::SoftmaxGradKernel<paddle::platform::CPUPlace, float>);
