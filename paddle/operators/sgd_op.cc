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

#include "paddle/operators/sgd_op.h"

namespace paddle {
namespace operators {

class SGDOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("param"),
                   "Input(param) of SGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("grad"),
                   "Input(grad) of SGDOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("param_out"),
                   "Output(param_out) of SGDOp should not be null.");

    auto param_dim = ctx->GetInputDim("param");
    PADDLE_ENFORCE_EQ(param_dim, ctx->GetInputDim("grad"),
                      "Two input of SGD Op's dimension must be same.");
    ctx->SetOutputDim("param_out", param_dim);
  }
};

class SGDOpMaker : public framework::OpInfoMaker {
 public:
  SGDOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpInfoMaker(proto, op_checker) {
    AddInput("param", "input parameter");
    AddInput("grad", "input gradient");
    AddOutput("param_out", "output parameter");
    AddAttr<float>("learning_rate", "learning rate of sgd");
    AddComment(R"DOC(

Simplest sgd algorithm.

param_out = param - learning_rate * grad;

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(sgd, ops::SGDOp, ops::SGDOpMaker);
REGISTER_OP_CPU_KERNEL(sgd,
                       ops::SGDOpKernel<paddle::platform::CPUPlace, float>);
