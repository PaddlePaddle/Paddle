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

#include "paddle/operators/rmsprop_op.h"

namespace paddle {
namespace operators {

class RmspropOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(param) of RmspropOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(grad) of RmspropOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Moment"),
                   "Input(moment) of RmspropOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(param_out) of RmspropOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("MomentOut"),
                   "Output(moment_out) of RmspropOp should not be null.");

    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("Grad"),
        "Param and grad input of RmspropOp should have the same dimension.");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("Moment"),
        "Param and moment input of RmspropOp should have the same dimension.");

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("MomentOut", param_dim);
  }
};

class RmspropOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RmspropOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Param", "Input parameter");
    AddInput("Grad", "Input gradient");
    AddInput("Moment", "Second moment");

    AddOutput("ParamOut", "Output parameter");
    AddOutput("MomentOut", "Output second moment");

    AddAttr<float>("learningRate", "Learning rate");
    AddAttr<float>("epsilon", "Constant for numerical stability");
    AddAttr<float>("decayRate", "Decay rate for moving average of gradients");
    AddComment(R"DOC(

RMSprop

MomentOut = decayRate * Moment + (1 - decayRate) * Grad * Grad
ParamOut = Param - learningRate * Grad / (sqrt(MomentOut) + epsilon)

The original slide(Slide 29 of
http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
does not have the epsilon attribute. It is added here for numerical stability
to avoid division by zero.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(rmsprop, ops::RmspropOp, ops::RmspropOpMaker);
REGISTER_OP_CPU_KERNEL(rmsprop,
                       ops::RmspropOpKernel<paddle::platform::CPUPlace, float>);
