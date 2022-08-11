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

#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
class Scope;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class BeamSearchDecodeOp : public framework::OperatorBase {
 public:
  BeamSearchDecodeOp(const std::string& type,
                     const framework::VariableNameMap& inputs,
                     const framework::VariableNameMap& outputs,
                     const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(dev_place);

    framework::RuntimeContext run_ctx(Inputs(), Outputs(), scope);
    framework::ExecutionContext ctx(*this, scope, dev_ctx, run_ctx);

    const LoDTensorArray* ids = ctx.Input<LoDTensorArray>("Ids");
    const LoDTensorArray* scores = ctx.Input<LoDTensorArray>("Scores");
    const size_t step_num = ids->size();
    PADDLE_ENFORCE_GT(
        step_num,
        0UL,
        platform::errors::InvalidArgument(
            "beam search steps, which is the"
            "size of Input(Ids) LoDTensorArray. beam search steps should "
            "be larger than 0, but received %d. ",
            step_num));

    const size_t source_num = ids->at(0).lod().at(0).size() - 1;
    PADDLE_ENFORCE_GT(
        source_num,
        0UL,
        platform::errors::InvalidArgument(
            "source_num is the sequence number of the"
            "first decoding step, indicating by Input(Ids)[0].lod[0].size. "
            "The number of source_num should be larger than"
            "0, but received %d. ",
            source_num));

    for (size_t i = 0; i < step_num; ++i) {
      PADDLE_ENFORCE_EQ(
          ids->at(i).lod().size(),
          2UL,
          platform::errors::InvalidArgument(
              "For the i step in beam search steps,"
              "the size of Input(Ids)[i].lod() should larger than 2,"
              "but received %d. ",
              ids->at(i).lod().size()));
    }

    size_t beam_size = ctx.Attr<int>("beam_size");
    int end_id = ctx.Attr<int>("end_id");

    // prepare output
    LoDTensor* sentenceIds = nullptr;
    LoDTensor* sentenceScores = nullptr;

    LoDTensor* sentenceIds_temp = ctx.Output<LoDTensor>("SentenceIds");
    LoDTensor* sentenceScores_temp = ctx.Output<LoDTensor>("SentenceScores");

    if (platform::is_xpu_place(ids->at(0).place())) {
      sentenceIds = new LoDTensor();
      sentenceIds->set_lod(sentenceIds_temp->lod());
    } else {
      sentenceIds = sentenceIds_temp;
    }

    if (platform::is_xpu_place(ids->at(0).place())) {
      sentenceScores = new LoDTensor();
      sentenceScores->set_lod(sentenceScores_temp->lod());
    } else {
      sentenceScores = sentenceScores_temp;
    }

    framework::VisitDataType(
        framework::TransToProtoVarType(scores->at(0).dtype()),
        BeamSearchDecodeFunctor(
            *ids, *scores, sentenceIds, sentenceScores, beam_size, end_id));

    if (platform::is_xpu_place(ids->at(0).place())) {
      int r = 0;
      r = CopyTensorByXPU<int64_t>(*sentenceIds, sentenceIds_temp, 1);
      PADDLE_ENFORCE_EQ(
          r,
          xpu::Error_t::SUCCESS,
          platform::errors::External(
              "Execute function CopyTensorByXPU failed by [%d]", r));

      r = CopyTensorByXPU<float>(*sentenceScores, sentenceScores_temp, 1);
      PADDLE_ENFORCE_EQ(
          r,
          xpu::Error_t::SUCCESS,
          platform::errors::External(
              "Execute function CopyTensorByXPU failed by [%d]", r));
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

class BeamSearchDecodeInferShape : public framework::InferShapeBase {
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
};

class BeamSearchDecodeInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
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

REGISTER_OPERATOR(
    beam_search_decode,
    paddle::operators::BeamSearchDecodeOp,
    paddle::operators::BeamSearchDecodeOpProtoMaker,
    paddle::operators::BeamSearchDecodeInferShape,
    paddle::operators::BeamSearchDecodeInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
