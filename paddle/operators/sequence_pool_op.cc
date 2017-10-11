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

#include "paddle/operators/sequence_pool_op.h"

namespace paddle {
namespace operators {

class SequencePoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequencePoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequencePoolOp should not be null.");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }
};

class SequencePoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequencePoolOpMaker(framework::OpProto* proto,
                      framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "A float LoDTensor, the variable-length input of SequencePoolOp");
    AddOutput(
        "Out",
        "A float LoDTensor, the variable-length output of SequencePoolOp.");
    AddAttr<int>(
        "strategy",
        "(int, default AVERAGE) the pooling strategy of SequencePoolOp.")
        .SetDefault(AVERAGE)
        .InEnum({AVERAGE, SUM, SQRT, MAX, LAST, FIRST});
    AddComment(R"DOC(
    SequencePoolOp pools features of all time-steps of each instance.

    For a mini-batch of 3 variable lengths sentences, containing 2, 3, and 2 time-steps:

    Assume X is a [7,M,N] float LoDTensor, and X->lod()[0] = [0, 2, 5, 7].
    Besides, for the sake of simplicity, we assume M=1 and N=1,
    and the value of X = [[1, 3], [2, 4, 6], [5, 1]].

    Thus, Out is a [3,1,1] float LoDTensor, but Out->lod() is nullptr.
    And for different strategy, the value of Out is as follows:

    - AVERAGE: [2, 4, 3], where 2=(1+3)/2, 4=(2+4+6)/3, 3=(5+1)/2
    - SUM: [4, 12, 6], where 4=1+3, 12=2+4+6, 6=5+1
    - SQRT: [2.82, 6.93, 4.24], where 2.82=(1+3)/sqrt(2),
           6.93=(2+4+6)/sqrt(3), 4.24=(5+1)/sqrt(2)
    - MAX: [3, 6, 5], where 3=max(1,3), 6=max(2,4,6), 5=max(5,1)
    - LAST: [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)
    - FIRST: [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)
    )DOC");
  }
};

class SequencePoolGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Gradient of Out should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("X"), "The input X should not be null.");
    auto og_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(og_dims.size(), x_dims.size(),
                      "The rank of output grad must equal to Input(X).");
    for (int64_t i = 1; i < og_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(og_dims[i], x_dims[i], "The dimension mismatch.");
    }
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sequence_pool, ops::SequencePoolOp, ops::SequencePoolOpMaker,
            sequence_pool_grad, ops::SequencePoolGradOp);
REGISTER_OP_CPU_KERNEL(
    sequence_pool, ops::SequencePoolKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    sequence_pool_grad,
    ops::SequencePoolGradKernel<paddle::platform::CPUPlace, float>);
