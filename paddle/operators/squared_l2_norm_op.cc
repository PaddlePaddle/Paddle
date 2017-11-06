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

#include "paddle/operators/squared_l2_norm_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class SquaredL2NormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should be not null.");

    ctx->SetOutputDim("Out", {1});
  }
};

class SquaredL2NormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

class SquaredL2NormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SquaredL2NormOpMaker(framework::OpProto* proto,
                       framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) The input of squared_l2_norm op.");
    AddOutput("Out", "(Scalar) The output of squared_l2_norm op.");
    AddComment(R"DOC(
SquaredL2Norm Operator.

Computes the squared L2 norm of a tensor.

$$Out = \sum_{i} X_{i}^2$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(squared_l2_norm, ops::SquaredL2NormOp, ops::SquaredL2NormOpMaker,
            squared_l2_norm_grad, ops::SquaredL2NormGradOp);
REGISTER_OP_CPU_KERNEL(
    squared_l2_norm,
    ops::SquaredL2NormKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    squared_l2_norm_grad,
    ops::SquaredL2NormGradKernel<paddle::platform::CPUPlace, float>);
