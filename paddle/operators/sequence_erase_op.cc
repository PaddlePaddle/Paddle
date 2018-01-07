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

#include "paddle/operators/sequence_erase_op.h"

namespace paddle {
namespace operators {

class SequenceEraseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequenceEraseOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequenceEraseOp should not be null.");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }
};

class SequenceEraseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequenceEraseOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(LoDTensor) 2-D input LoDTensor with the 2-nd dimension "
             "of length 1.");
    AddOutput("Out",
              "(LoDTensor) 2-D output LoDTensor with the 2-nd dimension "
              "of length 1.");
    AddAttr<std::vector<int>>("tokens",
                              "(vector<int>) "
                              "Tokens to be removed from input.");
    AddComment(R"DOC(
Sequence Erase Operator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(sequence_erase, ops::SequenceEraseOp,
                             ops::SequenceEraseOpMaker);
REGISTER_OP_CPU_KERNEL(
    sequence_erase,
    ops::SequenceEraseKernel<paddle::platform::CPUDeviceContext, int32_t>);
