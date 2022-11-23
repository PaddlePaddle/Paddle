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
    AddInput(
        "Emission",
        "(Tensor/LoDTensor). For a LoDTensor input, its shape is [N x D] "
        "where N is the total sequence length of the mini-batch and D is "
        "the total tag number. While for a tensor input, its shape is "
        "[B X S X D] with B the batch size and S the sequence length of each "
        "sample after padding. This input is the unscaled emission weight "
        "matrix of the linear_chain_crf operator. The data type is float32 "
        "or float64.");
    AddInput(
        "Transition",
        "(Tensor). A Tensor with shape [(D + 2) x D]. "
        "This input is the transition weights learned by the linear_chain_crf "
        "operator, denoted as w. The 1st row of w are transition weights for "
        "the start mask. The 2nd row of w are transition weights for the end "
        "mask. Transition weights between other tags begin from the 3rd row of "
        "w. See more details in comments of the linear_chain_crf operator. "
        "The data type is the same as Input(Emission).");
    AddInput(
        "Label",
        "(Tensor/LoDTensor). The ground truth with shape "
        "[N x 1] (for LoDTensor) or [B x S] (for Tensor). This input is "
        "optional. See more details in the operator's comments. The data type "
        "is int64.")
        .AsDispensable();
    AddOutput(
        "ViterbiPath",
        "(Tensor/LoDTensor). The decoding results. What to "
        "return changes depending on whether the Input(Label) (the ground "
        "truth) is given. See more details in the operator's comment. "
        "The data type is int64.");
    AddInput("Length",
             "(Tensor). The actual length of each sample before "
             "padding with shape [B x 1]. It means the Input(Emission), "
             "Input(Label) and Output(ViterbiPath) are common tensors with "
             "padding when this input is given. The data type is int64.")
        .AsDispensable();
    AddComment(R"DOC(
The crf_decoding operator reads the emission feature weights and the transition
feature weights learned by the linear_chain_crf operator and performs decoding.
It implements the Viterbi algorithm which is a dynamic programming algorithm
for finding the most likely sequence of hidden states, called the Viterbi path,
that results in a sequence of observed tags.

The output of this operator changes according to whether Input(Label) is given:

1. Input(Label) is given:
   This happens in training. This operator is used to co-work with the chunk_eval
   operator.
   When Input(Label) is given, the crf_decoding operator returns tensor with the
   sampe shape as Input(Label) whose values are fixed to be 0, indicating an
   incorrect prediction, or 1 indicating a tag is correctly predicted. Such an
   output is the input to chunk_eval operator.

2. Input(Label) is not given:
   This is the standard decoding process.

The crf_decoding operator returns a row vector with shape [N x 1]/[B x S], here
the shape depends on the inputs are LoDTensors or common tensors, whose values
range from 0 to maximum tag number - 1, Each element indicates an index of a
predicted tag.
)DOC");
  }
};

class CRFDecodingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Emission"), "Input", "Emission", "CRFDecoding");
    OP_INOUT_CHECK(
        ctx->HasInput("Transition"), "Input", "Transition", "CRFDecoding");
    OP_INOUT_CHECK(
        ctx->HasOutput("ViterbiPath"), "Output", "ViterbiPath", "CRFDecoding");

    auto emission_dims = ctx->GetInputDim("Emission");
    bool has_length = ctx->HasInput("Length");

    if (has_length) {
      PADDLE_ENFORCE_EQ(emission_dims.size(),
                        3,
                        platform::errors::InvalidArgument(
                            "The Input(Emission) should be a 3-D tensor. But "
                            "received: input rank %u, input shape [%s]. ",
                            emission_dims.size(),
                            emission_dims));
    } else {
      PADDLE_ENFORCE_EQ(emission_dims.size(),
                        2,
                        platform::errors::InvalidArgument(
                            "The Input(Emission) should be a 2-D tensor. But "
                            "received: input rank %u, input shape [%s].",
                            emission_dims.size(),
                            emission_dims));
    }

    auto transition_dims = ctx->GetInputDim("Transition");
    PADDLE_ENFORCE_EQ(transition_dims.size(),
                      2UL,
                      platform::errors::InvalidArgument(
                          "The Input(Transition) should be a 2-D tensor. But "
                          "received: input rank %u, input shape [%s].",
                          transition_dims.size(),
                          transition_dims));
    PADDLE_ENFORCE_EQ(
        transition_dims[0] - 2,
        transition_dims[1],
        platform::errors::InvalidArgument(
            "An invalid dimension for the Input(Transition), which should "
            "be a 2-D tensor with shape [(D + 2) x D]. But received: input "
            "rank %u, "
            "input shape [%s].",
            transition_dims.size(),
            transition_dims));
    if (ctx->IsRuntime() || (emission_dims[emission_dims.size() - 1] > 0 &&
                             transition_dims[transition_dims.size() - 1] > 0)) {
      PADDLE_ENFORCE_EQ(emission_dims[emission_dims.size() - 1],
                        transition_dims[transition_dims.size() - 1],
                        platform::errors::InvalidArgument(
                            "The last dimension of the Input(Emission) and the "
                            "Input(Transition) "
                            "should be equal to the tag number. But received "
                            "Input(Emission): rank "
                            "%u, shape [%s]; received Input(Transition): rank "
                            "%u, shape [%s].",
                            emission_dims.size(),
                            emission_dims,
                            transition_dims.size(),
                            transition_dims));
    }
    if (ctx->HasInput("Label")) {
      auto label_dims = ctx->GetInputDim("Label");
      if (ctx->HasInput("Length")) {
        PADDLE_ENFORCE_EQ(
            (label_dims.size() == 3UL && label_dims[2] == 1) ||
                label_dims.size() == 2UL,
            true,
            platform::errors::InvalidArgument(
                "The Input(Label) should be a 3-D tensor with last dimension "
                "fixed to 1 or a 2-D tensor in padding mode. But received: "
                "input "
                "rank %u, input shape [%s].",
                label_dims.size(),
                label_dims));
      } else {
        PADDLE_ENFORCE_EQ(
            (label_dims.size() == 2UL && label_dims[1] == 1) ||
                label_dims.size() == 1UL,
            true,
            platform::errors::InvalidArgument(
                "The Input(Label) should be a 2-D tensor with last "
                "dimension fixed to 1 or a 1-D tensor. But received: "
                "input rank %u, input shape [%s].",
                label_dims.size(),
                label_dims));
      }
      if (ctx->IsRuntime() || (emission_dims[0] > 0 && label_dims[0] > 0)) {
        PADDLE_ENFORCE_EQ(
            emission_dims[0],
            label_dims[0],
            platform::errors::InvalidArgument(
                "The first dimension of Input(Emission) and Input(Label) "
                "should be the same. But received Input(Emission): rank %u, "
                "shape [%s]; received Input(Label): rank %u, shape [%s].",
                emission_dims.size(),
                emission_dims,
                label_dims.size(),
                label_dims));
      }
    }

    ctx->ShareLoD("Emission", /*->*/ "ViterbiPath");
    if (has_length) {
      ctx->SetOutputDim("ViterbiPath", {emission_dims[0], emission_dims[1]});
    } else {
      ctx->SetOutputDim("ViterbiPath", {emission_dims[0], 1});
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Emission"),
        platform::CPUPlace());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(crf_decoding,
                             ops::CRFDecodingOp,
                             ops::CRFDecodingOpMaker);
REGISTER_OP_CPU_KERNEL(crf_decoding,
                       ops::CRFDecodingOpKernel<phi::CPUContext, float>,
                       ops::CRFDecodingOpKernel<phi::CPUContext, double>);
