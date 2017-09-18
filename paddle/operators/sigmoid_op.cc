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

#include "paddle/operators/sigmoid_op.h"

namespace paddle {
namespace operators {

class SigmoidOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input(X) of SigmoidOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Y"),
                            "Output(Y) of SigmoidOp should not be null.");

    ctx.Output<framework::LoDTensor>("Y")->Resize(
        ctx.Input<Tensor>("X")->dims());
  }
};

class SigmoidOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SigmoidOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "sigmoid input");
    AddOutput("Y", "sigmoid output");
    AddComment("Sigmoid function");
  }
};

class SigmoidOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    ctx.Output<framework::LoDTensor>(framework::GradVarName("X"))
        ->Resize(ctx.Input<Tensor>("Y")->dims());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sigmoid, ops::SigmoidOp, ops::SigmoidOpMaker, sigmoid_grad,
            ops::SigmoidOpGrad);
REGISTER_OP_CPU_KERNEL(sigmoid,
                       ops::SigmoidKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    sigmoid_grad, ops::SigmoidGradKernel<paddle::platform::CPUPlace, float>);
