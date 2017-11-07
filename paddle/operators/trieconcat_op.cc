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

#include "paddle/operators/trieconcat_op.h"

namespace paddle {
namespace operators {

class TrieConcatOp : public framework::OperatorBase {
 public:
  TrieConcatOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    framework::ExecutionContext ctx(*this, scope, dev_ctx);
    const std::vector<LoDTensor>* ids =
        ctx.Input<std::vector<LoDTensor>>("Ids");
    const std::vector<LoDTensor>* scores =
        ctx.Input<std::vector<LoDTensor>>("Scores");
    const size_t step_num = ids->size();
    PADDLE_ENFORCE_LT(step_num, 0, "beam search steps should be larger than 0");
    const size_t batch_size = ids->at(0).lod()[0].size() - 1;
    PADDLE_ENFORCE_LT(batch_size, 0UL, "batch size should be larger than 0");

    for (size_t i = 0; i < step_num; ++i) {
      PADDLE_ENFORCE_EQ(ids->at(i).lod().size(), 2UL,
                        "Level of LodTensor should be 2");
      //      PADDLE_ENFORCE_EQ(ids->at(i).lod(), scores->at(i).lod(),
      //                        "score and ids should have the same lod info");
    }

    // prepare output
    LoDTensor* sentenceIds = ctx.Output<LoDTensor>("SentenceIds");
    LoDTensor* sentenceScores = ctx.Output<LoDTensor>("SentenceScores");

    sentenceIds->Resize({kInitLength});
    sentenceIds->mutable_data<int64_t>(ids->at(0).place());
    sentenceScores->Resize({kInitLength});
    sentenceScores->mutable_data<float>(ids->at(0).place());

    BeamHelpter beam_helper;
    // init beam_nodes for each source sentence.
    std::vector<std::vector<BeamNode*>> batch_beam_nodes;
    batch_beam_nodes.reserve(batch_size);
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      std::vector<BeamNode*> beam_nodes;
      size_t batch_start = ids->at(0).lod()[0][batch_idx];
      size_t batch_end = ids->at(0).lod()[0][batch_idx + 1];
      for (size_t word_id_idx = batch_start; word_id_idx < batch_end;
           ++word_id_idx) {
        beam_nodes.push_back(
            new BeamNode(ids->at(0).data<int64_t>()[word_id_idx],
                         scores->at(0).data<float>()[word_id_idx]));
      }
      batch_beam_nodes.push_back(beam_nodes);
    }

    // pack all steps for one batch first, then another batch
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      for (size_t step_id = 1; step_id < step_num; ++step_id) {
        size_t batch_start = ids->at(step_id).lod()[0][batch_idx];
        std::vector<BeamNode*> result = beam_helper.PackTwoBeamStepOut(
            batch_start, batch_beam_nodes[batch_idx], ids->at(step_id),
            scores->at(step_id), sentenceIds, sentenceScores);
        batch_beam_nodes[batch_idx] = result;
      }

      // append last beam_node to result
      for (auto* beam_node : batch_beam_nodes[batch_idx]) {
        beam_helper.AppendBeamNodeToLoDTensor(beam_node, sentenceIds,
                                              sentenceScores);
      }

      // update batch_lod_level
      sentenceIds->mutable_lod()->at(0).push_back(sentenceIds->lod()[1].size());
      sentenceScores->mutable_lod()->at(0).push_back(
          sentenceScores->lod()[1].size());
    }
  }
};

class TrieConcatOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TrieConcatOpProtoMaker(framework::OpProto* proto,
                         framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Ids",
             "(vector<LodTensor>) "
             "score of the candidate words in each step");
    AddInput("Scores",
             "(vector<LodTensor>) "
             "score of the candidate words in each step");
    AddOutput("SentenceIds",
              "(LodTensor)"
              "All possible result sentences of word ids");
    AddOutput("SentenceScores",
              "(LodTensor)"
              ""
              "All possible result sentences of word scores");
    AddComment(R"DOC(
Pack the result of Beam search op into SentenceIds and SentenceScores.
)DOC");
  }
};

class TrieConcatInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    PADDLE_ENFORCE(context->HasInput("X"), "TrieConcatOp must has input X");
    PADDLE_ENFORCE(context->HasOutput("out"),
                   "TrieConcatOp must has output Out");
  }
};

class TrieConcatInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind& op_desc,
                  framework::BlockDescBind* block) const override {
    for (auto& o : op_desc.Output("Out")) {
      block->Var(o)->SetType(framework::VarDesc::LOD_TENSOR);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(trie_concat, paddle::operators::TrieConcatOp,
                  paddle::operators::TrieConcatOpProtoMaker,
                  paddle::operators::TrieConcatInferShape,
                  paddle::operators::TrieConcatInferVarType,
                  paddle::framework::EmptyGradOpMaker);
