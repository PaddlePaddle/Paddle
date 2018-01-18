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

#include "paddle/operators/sequence_reshape_op.h"

namespace paddle {
namespace operators {

class SequenceReshapeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequenceReshapeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequenceReshapeOp should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2U, "Rank of Input(X) should be 2.");
    int dimension = ctx->Attrs().Get<int>("dimension");
    ctx->SetOutputDim("Out", {{x_dims[0], static_cast<int64_t>(dimension)}});
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class SequenceReshapeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequenceReshapeOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "");
    AddOutput("Out", "");
    AddAttr<int>("dimension", "");
    AddAttr<bool>("is_padding", "Default padding zero.");
    AddComment(R"DOC()DOC");
  }
};

class SequenceReshapeGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput(framework::GradVarName("Out")),
        "Input(Out@GRAD) of SequenceReshapeGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Out"),
                   "Input(Out) of SequenceReshapeGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequenceReshapeGradOp should  not be null.");

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_reshape, ops::SequenceReshapeOp,
                  ops::SequenceReshapeOpMaker);
REGISTER_OPERATOR(sequence_reshape_grad, ops::SequenceReshapeGradOp);
REGISTER_OP_CPU_KERNEL(
    sequence_reshape,
    ops::SequenceReshapeKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    sequence_reshape_grad,
    ops::SequenceReshapeGradKernel<paddle::platform::CPUDeviceContext, float>);
