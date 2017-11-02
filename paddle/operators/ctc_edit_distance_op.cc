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

#include "paddle/operators/ctc_edit_distance_op.h"

namespace paddle {
namespace operators {

class CTCEditDistanceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X1"), "Input(X1) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("X2"), "Input(X2) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) shouldn't be null.");
    ctx->SetOutputDim("Out", {1});
  }
};

class CTCEditDistanceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CTCEditDistanceOpMaker(framework::OpProto *proto,
                         framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X1",
             "(2-D tensor with shape [M x 1]) The indices for "
             "hypothesis string");
    AddInput("X2",
             "(2-D tensor with shape [N x 1]) The indices "
             "for reference string.");
    AddAttr<bool>("normalized",
                  "(bool, default false) Indicated whether "
                  "normalize the Output(Out) by the length of reference "
                  "string (X2).")
        .SetDefault(false);
    AddOutput("Out",
              "(2-D tensor with shape [1 x 1]) "
              "The output distance of CTCEditDistance operator.");
    AddComment(R"DOC(

CTCEditDistance operator computes the edit distance of two sequences, one named
hypothesis and another named reference.

Edit distance measures how dissimilar two strings, one is hypothesis and another
is reference, are by counting the minimum number of operations to transform
one string into anthor.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(ctc_edit_distance, ops::CTCEditDistanceOp,
                             ops::CTCEditDistanceOpMaker);
REGISTER_OP_CPU_KERNEL(
    ctc_edit_distance,
    ops::CTCEditDistanceKernel<paddle::platform::CPUPlace, int32_t>,
    ops::CTCEditDistanceKernel<paddle::platform::CPUPlace, int64_t>);
