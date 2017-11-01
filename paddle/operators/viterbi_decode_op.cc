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

#include "paddle/operators/viterbi_decode_op.h"

namespace paddle {
namespace operators {
class ViterbiDecodeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ViterbiDecodeOpMaker(framework::OpProto* proto,
                       framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Emission", "");
    AddInput("Transition", "");
    AddInput("Label", "").AsDispensable();
    AddOutput("Viterbi", "");
    AddOutput("ViterbiScore", "");
    AddComment(R"DOC(
)DOC");
  }
};

class ViterbiDecodeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Emission"),
                   "Input(Emission) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Transition"),
                   "Input(Transition) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("Viterbi"),
                   "Output(Viterbi) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("ViterbiScore"),
                   "Output(ViterbiScore) should be not null.");

    auto emission_dims = ctx->GetInputDim("Emission");
    PADDLE_ENFORCE_EQ(emission_dims.size(), 2UL,
                      "The Input(Emission) should be a 2-D tensor.");
    PADDLE_ENFORCE(emission_dims[0], "An empty mini-batch is not allowed.");

    auto transition_dims = ctx->GetInputDim("Transition");
    PADDLE_ENFORCE_EQ(transition_dims.size(), 2UL,
                      "The Input(Transition) should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(
        transition_dims[0] - 2, transition_dims[1],
        "An invalid dimension for the Input(Transition), which should "
        "be a 2-D tensor with shape [(D + 2) x D].");
    PADDLE_ENFORCE_EQ(
        emission_dims[1], transition_dims[1],
        "The 2nd dimension of the Input(Emission) and the Input(Transition) "
        "should be equal to the tag number.");

    if (ctx->HasInput("Label")) {
      auto label_dims = ctx->GetInputDim("Label");
      PADDLE_ENFORCE(label_dims.size() == 2UL && label_dims[1] == 1UL,
                     "The Input(Label) should be a 2-D tensor with the 2nd "
                     "dimensions fixed to 1.");
      PADDLE_ENFORCE_EQ(
          emission_dims[0], label_dims[0],
          "The height of Input(Emission) and the height of Input(Label) "
          "should be the same.");
    }

    ctx->SetOutputDim("Viterbi", emission_dims);
    ctx->SetOutputDim("ViterbiScore", emission_dims);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(viterbi_decode, ops::ViterbiDecodeOp,
                             ops::ViterbiDecodeOpMaker);
REGISTER_OP_CPU_KERNEL(
    viterbi_decode,
    ops::ViterbiDecodeOpKernel<paddle::platform::CPUPlace, float>,
    ops::ViterbiDecodeOpKernel<paddle::platform::CPUPlace, double>);
