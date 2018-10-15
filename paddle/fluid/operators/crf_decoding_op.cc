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

#include "paddle/fluid/operators/crf_decoding_op.h"

namespace paddle {
namespace operators {
class CRFDecodingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Emission",
             "(LoDTensor, default: LoDTensor<float>). A LoDTensor with shape "
             "[N x D] where N is the size of the mini-batch and D is the total "
             "tag number. This input is the unscaled emission weight matrix of "
             "the linear_chain_crf operator.");
    AddInput(
        "Transition",
        "(Tensor, default: Tensor<float>). A Tensor with shape [(D + 2) x D]. "
        "This input is the transition weights learned by the linear_chain_crf "
        "operator, denoted as w. The 1st row of w are transition weights for "
        "the start mask. The 2nd row of w are transition weights for the end "
        "mask. Transition weights between other tags begin from the 3rd row of "
        "w. See more details in comments of the linear_chain_crf operator.");
    AddInput(
        "Label",
        "(LoDTensor,  LoDTensor<int64_t>). The ground truth with shape "
        "[N x 1]. This input is optional. See more details in the operator's "
        "comments.")
        .AsDispensable();
    AddOutput(
        "ViterbiPath",
        "(LoDTensor, LoDTensor<int64_t>). The decoding results. What to "
        "return changes depending on whether the Input(Label) (the ground "
        "truth) is given. See more details in the operator's comment.");
    AddComment(R"DOC(
The crf_decoding operator reads the emission feature weights and the transition
feature weights learned by the linear_chain_crf operator. It implements the
Viterbi algorithm which is a dynamic programming algorithm for finding the most
likely sequence of hidden states, called the Viterbi path, that results in a
sequence of observed tags.

The output of this operator changes according to whether Input(Label) is given:

1. Input(Label) is given:
   This happens in training. This operator is used to co-work with the chunk_eval
   operator.
   When Input(Label) is given, the crf_decoding operator returns a row vector
   with shape [N x 1] whose values are fixed to be 0, indicating an incorrect
   prediction, or 1 indicating a tag is correctly predicted. Such an output is the
   input to chunk_eval operator.

2. Input(Label) is not given:
   This is the standard decoding process.

The crf_decoding operator returns a row vector with shape [N x 1] whose values
range from 0 to maximum tag number - 1, Each element indicates an index of a
predicted tag.
)DOC");
  }
};

class CRFDecodingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Emission"),
                   "Input(Emission) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Transition"),
                   "Input(Transition) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("ViterbiPath"),
                   "Output(ViterbiPath) should be not null.");

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

    ctx->ShareLoD("Emission", /*->*/ "ViterbiPath");
    ctx->SetOutputDim("ViterbiPath", {emission_dims[0], 1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<LoDTensor>("Emission")->type()),
        platform::CPUPlace());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(crf_decoding, ops::CRFDecodingOp,
                             ops::CRFDecodingOpMaker);
REGISTER_OP_CPU_KERNEL(
    crf_decoding,
    ops::CRFDecodingOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CRFDecodingOpKernel<paddle::platform::CPUDeviceContext, double>);
