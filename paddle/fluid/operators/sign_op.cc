/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/sign_op.h"

namespace paddle {
namespace operators {

class SignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SignOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SignOp should not be null.");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

template <typename AttrType>
class SignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of sign operator.");
    AddOutput("Out", "(Tensor) Output tensor of sign operator.");
    AddComment(R"DOC(
Sign operator

$$Out = X.sign()$$
)DOC");
  }
};

class SignGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("scale");
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetAttr("scale", 0.0f);
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sign, ops::SignOp, ops::SignOpMaker<float>,
                  ops::SignGradMaker);
REGISTER_OP_CPU_KERNEL(
    sign, ops::SignKernel<paddle::platform::CPUDeviceContext, float>);
