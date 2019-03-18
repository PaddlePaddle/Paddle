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

#include "paddle/fluid/operators/sequence_ops/sequence_slice_op.h"

namespace paddle {
namespace operators {

class SequenceSliceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequenceSliceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Offset"),
                   "Input(Offset) of SequenceSliceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Length"),
                   "Input(Length) of SequenceSliceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequenceSliceOp should not be null.");
    auto input_dims = ctx->GetInputDim("X");

    auto offset_dim = ctx->GetInputDim("Offset");
    auto length_dim = ctx->GetInputDim("Length");

    PADDLE_ENFORCE_EQ(
        offset_dim.size(), 2UL,
        "Only support one level sequence now, The rank of offset must be 2.");
    PADDLE_ENFORCE_EQ(
        length_dim.size(), 2UL,
        "Only support one level sequence now, The rank of Length must be 2.");

    // Initialize the output's dims to maximum,
    // and re-set to real dims by the value of Offset and Length at kernel
    ctx->SetOutputDim("Out", input_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::LoDTensor>("X")->type(),
                                   ctx.device_context());
  }
};

class SequenceSliceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "The gradient of Out should not be null.");
    PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName("X")),
                   "The gradient of X should not be null.");
    ctx->SetOutputsDim(framework::GradVarName("X"), ctx->GetInputsDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::LoDTensor>("X")->type(),
                                   ctx.device_context());
  }
};

class SequenceSliceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor), "
             "the input of SequenceSliceOp.");
    AddInput("Offset",
             "(Tensor), "
             "a vector<int> to describe the offset of every input sequence for "
             "sub sequence item.");
    AddInput("Length",
             "(Tensor), "
             "a vector<int> to describe the length of every input sequence for "
             "sub sequence item.");
    AddOutput("Out", "(LoDTensor), the output of SequenceSliceOp.");
    AddComment(R"DOC(
Sequence slice operator

The operator crops a subsequence from given sequence with given start offset and subsequence length.
It only supports sequence (LoD Tensor with level number is 1).
- Case:
    X = [[a1, a2;
        b1, b2;
        c1, c2]
       [d1, d2;
        e1, e2]]
    LoD(X) = {{0, 3, 5}}; Dims(X) = (5, 2)
    Offset = [[0], [1]]; Length = [[2], [1]]

    Out = [[a1, a2;
            b1, b2]
            [e1, e2]]
    LoD(Out) = {{0, 2, 3}}; Dims(Out) = (3, 2)
NOTE: The first dimension size of input, the size of offset and Length, should be equal. The offset start from 0.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_slice, ops::SequenceSliceOp,
                  ops::SequenceSliceOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(sequence_slice_grad, ops::SequenceSliceGradOp);
REGISTER_OP_CPU_KERNEL(
    sequence_slice,
    ops::SequenceSliceOpKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    sequence_slice_grad,
    ops::SequenceSliceGradOpKernel<paddle::platform::CPUDeviceContext, float>);
