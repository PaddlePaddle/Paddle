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

#include "paddle/operators/fill_zeros_like_op.h"

namespace paddle {
namespace operators {

class FillZerosLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of FillZerosLikeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"),
                   "Output(Y) of FillZerosLikeOp should not be null.");
    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Y");
  }
};

class FillZerosLikeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FillZerosLikeOpMaker(framework::OpProto *proto,
                       framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of fill-zeros-like op.");
    AddOutput("Y", "The variable will be filled up with zeros.");
    AddComment(R"DOC(
FillZerosLike Operator.

Fill up a variable with zeros.
The output will have the same size as the input.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fill_zeros_like, ops::FillZerosLikeOp,
                             ops::FillZerosLikeOpMaker);
REGISTER_OP_CPU_KERNEL(
    fill_zeros_like,
    ops::FillZerosLikeKernel<paddle::platform::CPUPlace, float>);
