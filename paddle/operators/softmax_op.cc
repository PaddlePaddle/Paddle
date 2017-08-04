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

class SoftmaxOp : public OperatorWithKernel {
protected:
  void InferShape(const InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(ctx.InputSize() == 1UL,
                   "Only one input is need for softmax");
    PADDLE_ENFORCE(ctx.Input<Tensor>("X")->dims().size() == 2UL,
                   "The input of softmax op must be matrix");
    PADDLE_ENFORCE(ctx.OutputSize() == 1UL,
                   "Only one output is need for softmax");
    ctx.Output<Tensor>("Y")->Resize(ctx.Input<Tensor>("X")->dims());
  }
};

class SoftmaxOpMaker : public OpProtoAndCheckerMaker {
public:
  SoftmaxOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "input of softmax");
    AddOutput("Y", "output of softmax");
    AddComment("Softmax Op");
  }
};

class SoftmaxOpGrad : public OperatorWithKernel {
protected:
  void InferShape(const InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(ctx.InputSize() == 3UL,
                   "Input of SoftmaxOpGrad should be 3, X, Y, YG");
    PADDLE_ENFORCE(ctx.OutputSize() == 1UL,
                   "Output of SoftmaxOpGrad should be 1");
    PADDLE_ENFORCE(ctx.InputVar("Y") != nullptr, "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx.InputVar(framework::GradVarName("Y")) != nullptr,
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

REGISTER_OP(softmax, ops::SoftmaxOp, ops::SoftmaxOpMaker);
REGISTER_OP_CPU_KERNEL(softmax, ops::SoftmaxKernel<ops::CPUPlace, float>);
REGISTER_GRADIENT_OP(softmax, softmax_grad, ops::SoftmaxOpGrad);
REGISTER_OP_CPU_KERNEL(softmax_grad,
                       ops::SoftmaxGradKernel<ops::CPUPlace, float>);
