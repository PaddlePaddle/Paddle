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

#include "paddle/operators/adam_op.h"

namespace paddle {
namespace operators {

class AdamOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("param"),
        "Input(param) of AdamOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("grad"),
        "Input(grad) of AdamOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("moment1"),
        "Input(moment1) of AdamOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("moment2"),
        "Input(moment2) of AdamOp should not be null.");

    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("param_out"),
        "Output(param_out) of AdamOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("moment1_out"),
        "Output(moment1_out) of AdamOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("moment2_out"),
        "Output(moment2_out) of AdamOp should not be null.");

    PADDLE_ENFORCE_EQ(ctx.Input<Tensor>("param")->dims(),
                      ctx.Input<Tensor>("grad")->dims(),
                      "Two input of Adam Op's dimension must be same.");
    PADDLE_ENFORCE_EQ(ctx.Input<Tensor>("grad")->dims(),
                      ctx.Input<Tensor>("moment1")->dims(),
                      "Two input of Adam Op's dimension must be same.");
    PADDLE_ENFORCE_EQ(ctx.Input<Tensor>("moment1")->dims(),
                      ctx.Input<Tensor>("moment2")->dims(),
                      "Two input of Adam Op's dimension must be same.");

    ctx.Output<framework::Tensor>("param_out")
        ->Resize(ctx.Input<Tensor>("param")->dims());
    ctx.Output<framework::Tensor>("moment1_out")
        ->Resize(ctx.Input<Tensor>("moment1")->dims());
    ctx.Output<framework::Tensor>("moment2_out")
        ->Resize(ctx.Input<Tensor>("moment2")->dims());
  }
};

class AdamOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AdamOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("param", "input parameter");
    AddInput("grad", "input gradient");
    AddInput("moment1", "first moment");
    AddInput("moment2", "second moment");

    AddOutput("param_out", "output parameter");
    AddOutput("moment1_out", "output first moment");
    AddOutput("moment2_out", "output second moment");

    AddAttr<int>("time_step", "time step");
    AddAttr<float>("learning_rate", "learning rate of sgd");
    AddAttr<float>("epsilon", "prevent divide by zero");
    AddAttr<float>("beta1", "exponential decay for the first moment");
    AddAttr<float>("beta2", "exponential decay for the second moment");
    AddComment(R"DOC(

Adam Gradient Descent algorithm.

moment1_out = beta1 * moment1 + (1 − beta1) * grad
moment2_out = beta2 * moment2 + (1 − beta2) * grad * grad
moment1_hat =  moment1_out / (1 - beta1^t)
moment2_hat =  moment2_out / (1 - beta2^t)
param_out = param - learning_rate * moment1_hat / (sqrt(moment2_hat) + epsilon)

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adam, ops::AdamOp, ops::AdamOpMaker);
REGISTER_OP_CPU_KERNEL(adam,
                       ops::AdamOpKernel<paddle::platform::CPUPlace, float>);
