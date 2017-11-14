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

#include "paddle/operators/sequence_slice_op.h"

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

    ctx->SetOutputDim("Out", input_dims);
    }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
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
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
        ctx.device_context());
  }
};

class SequenceSliceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequenceSliceOpMaker(framework::OpProto* proto,
                       framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(LoDTensor), "
             "the input of SequenceSliceOp.");
    AddInput("Offset",
             "(Tensor), "
             "A vector<int> to describes offset for sub sequence item.");
    AddInput("Length",
             "(Tensor), "
             "A vector<int> to describes length for sub sequence item.");
    AddOutput("Out",
              "(LoDTensor), output of sequence slice Op.");
    AddComment(R"DOC(
Sequence slice operator
The operator crop a subsequence from given sequence with given start offset and subsequence length.
It only supports sequence (LoD Tensor with level number is 1).
- Case:
    X = [[a1, a2;
        b1, b2;
        c1, c2]
       [d1, d2;
        e1, e2]]
    LoD(X) = {{0, 3, 5}}; Dims(X) = (4, 1, 2)
    Offset = (0, 1); Length = (2, 1)

    Out = [[a1, a2;
            b1, b2]
            [e1, e2]]
    LoD(Out) = {{0, 2, 3}}
NOTE: The length of the input, offset and length should be the same. The offset start from 0.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sequence_slice, ops::SequenceSliceOp, ops::SequenceSliceOpMaker,
            sequence_slice_grad, ops::SequenceSliceGradOp);
REGISTER_OP_CPU_KERNEL(
    sequence_slice,
    ops::SequenceSliceOpKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    sequence_slice_grad,
    ops::SequenceSliceGradOpKernel<paddle::platform::CPUPlace, float>);
