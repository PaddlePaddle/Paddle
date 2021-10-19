/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/viterbi_decode_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class ViterbiDecodeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "ViterbiDecode");
    OP_INOUT_CHECK(ctx->HasInput("Transition"), "Input", "Transition",
                   "ViterbiDecode");
    OP_INOUT_CHECK(ctx->HasInput("Length"), "Input", "Length", "ViterbiDecode");
    OP_INOUT_CHECK(ctx->HasOutput("Scores"), "Output", "Scores",
                   "ViterbiDecode");
    OP_INOUT_CHECK(ctx->HasOutput("Path"), "Output", "Path", "ViterbiDecode");
    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_EQ(in_dims.size(), 3,
                      platform::errors::InvalidArgument(
                          "The rank of Input in ViterbiDecode  must be 3. But "
                          "received Input's rank is %d.",
                          in_dims.size()));
    auto length_dims = ctx->GetInputDim("Length");
    PADDLE_ENFORCE_EQ(length_dims.size(), 1,
                      platform::errors::InvalidArgument(
                          "The rank of Length in ViterbiDecode must be 1. But "
                          "received Length's rank is %d.",
                          length_dims.size()));
    auto transition_dims = ctx->GetInputDim("Transition");
    PADDLE_ENFORCE_EQ(
        transition_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The rank of Transition in ViterbiDecode must be 2. But "
            "received Transition's rank is %d.",
            transition_dims.size()));
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          in_dims[0], length_dims[0],
          platform::errors::InvalidArgument(
              "The batch size of Input and Length should be equal."));
      PADDLE_ENFORCE_EQ(in_dims[2], transition_dims[0],
                        platform::errors::InvalidArgument(
                            "The number of tags of Input (%d) and Transition "
                            "(%d) should be equal.",
                            transition_dims[0], in_dims[2]));
    }
    ctx->SetOutputDim("Scores", length_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class ViterbiDecodeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "The unary emission tensor. The shape of Input must be (batch_size,"
        "sequence_length, num_tags). ");
    AddInput("Transition",
             "The transition matrix. The shape of Transition must be ( "
             "num_tags, num_tags). ");
    AddInput("Length",
             "The input length tensor storing real length of each sequence for "
             "correctness. The shape of Length MUST be (batch_size).");
    AddOutput("Scores",
              "The scores tensor containing the score for the Viterbi "
              "sequence. The shape of Scores MUST be (batch_size).");
    AddOutput("Path",
              "The paths tensor containing the highest scoring tag indices. "
              "The shape of Scores MUST be (batch_size, sequence_length).");
    AddAttr<bool>("include_bos_eos_tag",
                  "If set to True, the last row and the last column of "
                  "transitions will be considered as start tag.")
        .SetDefault(true);
    AddComment(R"DOC(
      )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace platform = paddle::platform;
REGISTER_OP_WITHOUT_GRADIENT(viterbi_decode, ops::ViterbiDecodeOp,
                             ops::ViterbiDecodeOpMaker);
REGISTER_OP_CPU_KERNEL(
    viterbi_decode, ops::ViterbiDecodeKernel<platform::CPUDeviceContext, float>,
    ops::ViterbiDecodeKernel<platform::CPUDeviceContext, double>);
