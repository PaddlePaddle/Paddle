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

#include "paddle/operators/sub_sequence_op.h"

namespace paddle {
namespace operators {

class SubSequenceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SubSequenceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SubSequenceOp should not be null.");
    auto input_dims = ctx->GetInputDim("X");

    auto offsets = ctx->Attrs().Get<std::vector<int>>("offset");
    auto sizes = ctx->Attrs().Get<std::vector<int>>("size");

    auto dim_0 = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
      dim_0 += sizes[i];
    }

    framework::DDim out_dims = input_dims;
    out_dims[0] = dim_0;
    ctx->SetOutputDim("Out", out_dims);
  }
};

class SubSequenceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "The gradient of Out should not be null.");
    PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName("X")),
                   "The gradient of X should not be null.");
    ctx->SetOutputsDim(framework::GradVarName("X"), ctx->GetInputsDim("X"));
  }
};

class SubSequenceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SubSequenceOpMaker(framework::OpProto* proto,
                        framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(LoDTensor), "
            "the variable-length input of SubSequenceOp");
    AddAttr<std::vector<int>>(
        "offset",
        "A list<int> to describes offset for sub sequence item.");
    AddAttr<std::vector<int>>(
        "size",
        "A list<int> to describes size for sub sequence item.");
    AddOutput("Out",
              "(Tensor), Variable-length output of "
              "sequence_concat Op.");
    AddComment(R"DOC(
Sub Sequence operator
          
The operator crop a subsequence from given sequence with given start offset and subsequence size.
It only supports sequence (LoD Tensor with level number is 1).
- Case:
    LoD(x) = {{0, 3, 6, 10}}; Dims(x0) = (10, 3, 2)
    offset = (0, 1, 1); size = (2, 1, 2)
    LoD(Out) = {{0, 2, 3, 5}}; Dims(Out) = (5,3,2)
NOTE: The length of the input, offset and size should be the same. The offset start from 0.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sub_sequence, ops::SubSequenceOp, ops::SubSequenceOpMaker,
            sub_sequence_grad, ops::SubSequenceGradOp);
REGISTER_OP_CPU_KERNEL(
    sub_sequence,
    ops::SubSequenceOpKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    sub_sequence_grad,
    ops::SubSequenceGradOpKernel<paddle::platform::CPUPlace, float>);
