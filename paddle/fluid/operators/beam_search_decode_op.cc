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

#include "paddle/fluid/operators/beam_search_decode_op.h"
#include <string>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class BeamSearchDecodeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    for (const std::string &arg : std::vector<std::string>({"Ids", "Scores"})) {
      OP_INOUT_CHECK(ctx->HasInput(arg), "Input", arg, "BeamSeachDecode");
    }
    for (const std::string &arg :
         std::vector<std::string>({"SentenceIds", "SentenceScores"})) {
      OP_INOUT_CHECK(ctx->HasOutput(arg), "Output", arg, "BeamSeachDecode");
    }
  }
};

class BeamSearchDecodeOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "(LodTensorArray)"
             "The LodTensorArray containing the selected ids of all steps");
    AddInput("Scores",
             "(LodTensorArray)"
             "The LodTensorArray containing the selected scores of all steps");
    AddOutput(
        "SentenceIds",
        "(LodTensor)"
        "An LodTensor containing all generated id sequences for all source "
        "sentences");
    AddOutput(
        "SentenceScores",
        "(LodTensor)"
        "An LodTensor containing scores corresponding to Output(SentenceIds)");
    AddAttr<int>("beam_size", "beam size for beam search");
    AddAttr<int>("end_id",
                 "the token id which indicates the end of a sequence");
    AddComment(R"DOC(
Beam Search Decode Operator. This Operator constructs the full hypotheses for
each source sentence by walking back along the LoDTensorArray Input(ids)
whose lods can be used to restore the path in the beam search tree.

The Output(SentenceIds) and Output(SentenceScores) separately contain the
generated id sequences and the corresponding scores. The shapes and lods of the
two LodTensor are same. The lod level is 2 and the two levels separately
indicate how many hypotheses each source sentence has and how many ids each
hypothesis has.
)DOC");
  }
};

/*class BeamSearchDecodeInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(
        context->HasInput("Ids"), "Input", "Ids", "BeamSearchDecode");
    OP_INOUT_CHECK(
        context->HasInput("Scores"), "Input", "Scores", "BeamSearchDecode");
    OP_INOUT_CHECK(context->HasOutput("SentenceIds"),
                   "Output",
                   "SentenceIds",
                   "BeamSearchDecode");
    OP_INOUT_CHECK(context->HasOutput("SentenceScores"),
                   "Output",
                   "SentenceScores",
                   "BeamSearchDecode");
  }
};*/

class BeamSearchDecodeInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SetOutputType("SentenceIds",
                       framework::proto::VarType::LOD_TENSOR,
                       framework::ALL_ELEMENTS);
    ctx->SetOutputType("SentenceScores",
                       framework::proto::VarType::LOD_TENSOR,
                       framework::ALL_ELEMENTS);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(beam_search_decode,
                  paddle::operators::BeamSearchDecodeOp,
                  paddle::operators::BeamSearchDecodeOpProtoMaker,
                  paddle::operators::BeamSearchDecodeInferVarType);

REGISTER_OP_CPU_KERNEL(
    beam_search_decode,
    ops::BeamSearchDecodeOpKernel<phi::CPUContext, float>,
    ops::BeamSearchDecodeOpKernel<phi::CPUContext, double>,
    ops::BeamSearchDecodeOpKernel<phi::CPUContext, paddle::platform::float16>,
    ops::BeamSearchDecodeOpKernel<phi::CPUContext, int>,
    ops::BeamSearchDecodeOpKernel<phi::CPUContext, int64_t>);
