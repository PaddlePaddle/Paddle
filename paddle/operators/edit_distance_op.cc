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

#include "paddle/operators/edit_distance_op.h"

namespace paddle {
namespace operators {

class EditDistanceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Hyp"), "Input(Hyp) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("Ref"), "Input(Ref) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) shouldn't be null.");
    ctx->SetOutputDim("Out", {1});
  }

 protected:
  framework::OpKernelType GetActualKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(framework::proto::DataType::FP32,
                                   ctx.device_context());
  }
};

class EditDistanceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  EditDistanceOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Hyp",
             "(2-D tensor with shape [M x 1]) The indices for "
             "hypothesis string");
    AddInput("Ref",
             "(2-D tensor with shape [N x 1]) The indices "
             "for reference string.");
    AddAttr<bool>("normalized",
                  "(bool, default false) Indicated whether "
                  "normalize the Output(Out) by the length of reference "
                  "string (Ref).")
        .SetDefault(false);
    AddOutput("Out",
              "(2-D tensor with shape [1 x 1]) "
              "The output distance of EditDistance operator.");
    AddComment(R"DOC(

EditDistance operator computes the edit distance of two sequences, one named
hypothesis with length M and another named reference with length N.

Edit distance, also called Levenshtein distance, measures how dissimilar two strings 
are by counting the minimum number of operations to transform one string into anthor. 
Here the operations include insertion, deletion, and substitution. For example, 
given hypothesis string A = "kitten" and reference B = "sitting", the edit distance 
is 3 for A will be transformed into B at least after two substitutions and one 
insertion:
  
   "kitten" -> "sitten" -> "sittin" -> "sitting"

If Attr(normalized) is true, the edit distance will be divided by the length of 
reference string N.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(edit_distance, ops::EditDistanceOp, ops::EditDistanceOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    edit_distance, ops::EditDistanceKernel<paddle::platform::CPUPlace, float>);
