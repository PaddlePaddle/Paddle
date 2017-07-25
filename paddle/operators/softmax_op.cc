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
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

class SoftmaxOp : public framework::OperatorWithKernel {
protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(ctx.InputSize() == 1, "Only one input is need for softmax");
    PADDLE_ENFORCE(ctx.Input(0).dims().size() == 2,
                   "The input of softmax op must be matrix");
    PADDLE_ENFORCE(ctx.OutputSize() == 1,
                   "Only one output is need for softmax");
    ctx.Output(0)->Resize(ctx.Input(0).dims());
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
protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {}
  std::string DebugString() const override {
    LOG(INFO) << "SoftmaxOpGrad";
    return "";
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(softmax, ops::SoftmaxOp, ops::SoftmaxOpMaker);
REGISTER_GRADIENT_OP(softmax, softmax_grad, paddle::operators::SoftmaxOpGrad);
REGISTER_OP_CPU_KERNEL(softmax,
                       ops::SoftmaxKernel<paddle::platform::CPUPlace, float>);
