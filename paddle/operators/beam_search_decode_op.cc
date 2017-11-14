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

#include "paddle/operators/beam_search_decode_op.h"

namespace paddle {
namespace operators {

class BeamSearchDecodeOp : public framework::OperatorBase {
 public:
  BeamSearchDecodeOp(const std::string& type,
                     const framework::VariableNameMap& inputs,
                     const framework::VariableNameMap& outputs,
                     const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    framework::ExecutionContext ctx(*this, scope, dev_ctx);
    const LoDTensorArray* ids = ctx.Input<LoDTensorArray>("Ids");
    const LoDTensorArray* scores = ctx.Input<LoDTensorArray>("Scores");
    const size_t step_num = ids->size();
    PADDLE_ENFORCE_GT(step_num, 0UL,
                      "beam search steps should be larger than 0");
    const size_t source_num = ids->at(0).lod().at(0).size() - 1;
    PADDLE_ENFORCE_GT(source_num, 0UL, "source num should be larger than 0");

    for (size_t i = 0; i < step_num; ++i) {
      PADDLE_ENFORCE_EQ(ids->at(i).lod().size(), 2UL,
                        "Level of LodTensor should be 2");
    }

    // prepare output
    LoDTensor* sentenceIds = ctx.Output<LoDTensor>("SentenceIds");
    LoDTensor* sentenceScores = ctx.Output<LoDTensor>("SentenceScores");

    BeamSearchDecoder<float> beam_search_decoder;
    beam_search_decoder.PackAllSteps(*ids, *scores, sentenceIds,
                                     sentenceScores);
  }
};

class BeamSearchDecodeOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BeamSearchDecodeOpProtoMaker(framework::OpProto* proto,
                               framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Ids",
             "(LodTensorArray)"
             "score of the candidate words in each step");
    AddInput("Scores",
             "(LodTensorArray)"
             "score of the candidate words in each step");
    AddOutput("SentenceIds",
              "(LodTensor)"
              "All possible result sentences of word ids");
    AddOutput("SentenceScores",
              "(LodTensor)"
              "All possible result sentences of word scores");
    AddComment(R"DOC(
Pack the result of Beam search op into SentenceIds and SentenceScores.
)DOC");
  }
};

class BeamSearchDecodeInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    PADDLE_ENFORCE(context->HasInput("Ids"),
                   "BeamSearchDecodeOp must has input Ids");
    PADDLE_ENFORCE(context->HasInput("Scores"),
                   "BeamSearchDecodeOp must has input Scores");
    PADDLE_ENFORCE(context->HasOutput("SentenceIds"),
                   "BeamSearchDecodeOp must has output SentenceIds");
    PADDLE_ENFORCE(context->HasOutput("SentenceScores"),
                   "BeamSearchDecodeOp must has output SentenceScores");
  }
};

class BeamSearchDecodeInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind& op_desc,
                  framework::BlockDescBind* block) const override {
    for (auto& o : op_desc.Output("SentenceIds")) {
      block->Var(o)->SetType(framework::VarDesc::LOD_TENSOR);
    }
    for (auto& o : op_desc.Output("SentenceScores")) {
      block->Var(o)->SetType(framework::VarDesc::LOD_TENSOR);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(beam_search_decode, paddle::operators::BeamSearchDecodeOp,
                  paddle::operators::BeamSearchDecodeOpProtoMaker,
                  paddle::operators::BeamSearchDecodeInferShape,
                  paddle::operators::BeamSearchDecodeInferVarType,
                  paddle::framework::EmptyGradOpMaker);
