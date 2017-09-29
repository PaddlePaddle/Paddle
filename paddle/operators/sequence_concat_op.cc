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

#include "paddle/operators/sequence_concat_op.h"

namespace paddle {
namespace operators {

class SequenceConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE_GT(ctx->Inputs("X").size(), 0UL,
                      "Inputs(X) of SequenceConcatOp should not be empty.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequenceConcatOp should not be null.");
    const size_t level = static_cast<size_t>(ctx->Attrs().Get<int>("level"));
    const size_t axis = static_cast<size_t>(ctx->Attrs().Get<int>("axis"));
    PADDLE_ENFORCE(level == 0UL || level == 1UL,
                   "Sequence Concat Op only support one or two sequence now.");
    auto ins_dims = ctx->GetInputsDim("X");
    framework::DDim out_dims = ins_dims[0];
    const size_t n = ins_dims.size();
    for (size_t i = 1; i < n; i++) {
      out_dims[axis] += ins_dims[i][axis];
    }
    ctx->SetOutputDim("Out", out_dims);
  }
};

class SequenceConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequenceConcatOpMaker(framework::OpProto* proto,
                        framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "Multip LodTensors, the variable-length inputs of "
             "SequenceConcatOp")
        .AsDuplicable();
    AddOutput("Out",
              "A float LodTensor, the variable-length output of "
              "SequenceConcatOp.");
    AddAttr<int>("axis",
                 "The axis which the inputs will be joined with."
                 "If axis is 0, the inputs will be joined with Lod index.")
        .SetDefault(0);
    AddAttr<int>("level",
                 "The level which the inputs will be joined with."
                 "If level is 0, the inputs will be joined with word."
                 "If level is 1, the inputs will be joined with sentence.")
        .SetDefault(0);
    AddComment(R"DOC(
    SequenceConcatOp concat multip LodTensors and only supports one or two levels.
    - Case1:
      axis is 1, level is 1, the Lod of Inputs are the same,
      LoD(x0) = {{0,2,4},{0,1,2,3,4}}; Dims(x0) = (2,3,4)
      LoD(x1) = {{0,2,4},{0,1,2,3,4}}; Dims(x1) = (2,4,4)
      LoD(Out) = {{0,2,4},{01,2,3,4}}; Dims(Out) = (2,7,4)
    - Case2:
      If axis is 0, level is 1, the Lod of inputs are different,
      LoD(x0) = {{0,2,4}, {0,1,2,3,4}}; Dims(x0) = (2,3,4)
      LoD(x1) = {{0,3,5}, {0,1,3,4,5}}; Dims(x1) = (3,3,4)
      LoD(Out) = {{0,5,9}, {0,1,2,4,5,6,7,8,9}}; Dims(Out) = (5,3,4)
    )DOC");
  }
};

class SequenceConcatGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Gradient of Out should not be null.");
    PADDLE_ENFORCE_GT(ctx->Outputs(framework::GradVarName("X")).size(), 0UL,
                      "Gradient of X should not be empty.")
    ctx->SetOutputsDim(framework::GradVarName("X"), ctx->GetInputsDim("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sequence_concat, ops::SequenceConcatOp, ops::SequenceConcatOpMaker,
            sequence_concat_grad, ops::SequenceConcatGradOp);
REGISTER_OP_CPU_KERNEL(
    sequence_concat,
    ops::SequenceConcatOpKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    sequence_concat_grad,
    ops::SequenceConcatGradOpKernel<paddle::platform::CPUPlace, float>);
