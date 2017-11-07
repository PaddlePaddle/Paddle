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

using value_type = float;
using LoDTensor = framework::LoDTensor;

const int64_t kInitLength = 1024;
const int64_t kEndId = 0;

struct BeamNode {
  BeamNode(int64_t word_id, float prob) : word_id_(word_id), prob_(prob) {}

  BeamNode* father_ = nullptr;
  std::vector<BeamNode*> kids_;
  int64_t word_id_;
  float prob_;
};

void RemoveFromEnd(BeamNode* end) {
  PADDLE_ENFORCE_EQ(end->kids_.size(), 0UL, "end should not have any kids");
  auto* father = end->father_;
  if (father != nullptr) {
    auto kids = father->kids_;
    kids.erase(std::remove(kids.begin(), kids.end(), end), kids.end());
    delete end;
    if (father->kids_.size() == 0) {
      RemoveFromEnd(father);
    }
  }
}

struct AppendBeamNodeToLoDTensor {
  template <typename T>
  void AppendVector(const std::vector<T> data, LoDTensor* dst) {
    std::vector<size_t> sentances = dst->lod()[1];
    // TODO(check)
    T* dst_data = dst->data<T>() + sentances[sentances.size() - 1];
    memcpy(dst_data, &data, data.size() * sizeof(int));
    dst->mutable_lod()->at(0).push_back(sentances.back() + data.size());
  }

  void operator()(BeamNode* node, LoDTensor* dst_ids, LoDTensor* dst_probs) {
    std::vector<int64_t> sequence_ids;
    std::vector<float> sequence_probs;
    BeamNode* tmp = node;
    while (tmp != nullptr) {
      sequence_ids.push_back(tmp->word_id_);
      sequence_probs.push_back(tmp->prob_);
      tmp = tmp->father_;
    }
    AppendVector<int64_t>(sequence_ids, dst_ids);
    AppendVector<float>(sequence_probs, dst_probs);
  }
};

class PackTwoBeamStepOut {
 public:
  std::vector<BeamNode*> operator()(size_t batch_start,
                                    const std::vector<BeamNode*>& pre_results,
                                    const LoDTensor& cur_ids,
                                    const LoDTensor& cur_probs,
                                    LoDTensor* result_seq_ids,
                                    LoDTensor* result_probs) {
    //    PADDLE_ENFORCE_EQ(cur_ids.lod(), cur_probs.lod(),
    //                      "lod of ids and probs should be the same");
    std::vector<BeamNode*> result;
    std::vector<size_t> candidate_offset = cur_ids.lod()[0];
    for (size_t i = 0; i < pre_results.size(); ++i) {
      size_t candidate_start = candidate_offset[batch_start + i];
      size_t candidate_end = candidate_offset[batch_start + i + 1];
      if (candidate_start == candidate_end) {
        VLOG(3) << "this prefix does not have candidate";
        auto* prefix_end = pre_results[i];
        if (prefix_end->word_id_ == kEndId) {
          VLOG(3) << "find an end Id, append to result tensor";
          AppendBeamNodeToLoDTensor appender;
          appender(prefix_end, result_seq_ids, result_probs);
        } else {
          VLOG(3) << "this sentence has no more candidate, prune it";
        }
        // remove from Beam Tree
        RemoveFromEnd(prefix_end);
      } else {
        for (size_t candidate_index = candidate_start;
             candidate_index < candidate_end; ++candidate_index) {
          int64_t word_id = cur_ids.data<int64_t>()[candidate_index];
          PADDLE_ENFORCE_NE(word_id, kEndId,
                            "End id should not have candidate anymore");
          float prob = cur_probs.data<float>()[candidate_index];
          auto* candidate = new BeamNode(word_id, prob);
          auto* prefix = pre_results[i];
          candidate->father_ = prefix;
          prefix->kids_.push_back(candidate);
          result.push_back(candidate);
        }
      }
    }
    return result;
  }
};

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

    // beam_nodes for each source sentence.
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
      batch_beam_nodes[batch_idx] = beam_nodes;
    }

    // pack all steps for one batch first, then another batch
    PackTwoBeamStepOut packer;
    AppendBeamNodeToLoDTensor appender;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      for (size_t step_id = 1; step_id < step_num; ++step_id) {
        size_t batch_start = ids->at(step_id).lod()[0][batch_idx];
        std::vector<BeamNode*> result =
            packer(batch_start, batch_beam_nodes[batch_idx], ids->at(step_id),
                   scores->at(step_id), sentenceIds, sentenceScores);
        batch_beam_nodes[batch_idx] = result;
      }

      // append last beam_node to result
      for (auto* beam_node : batch_beam_nodes[batch_idx]) {
        appender(beam_node, sentenceIds, sentenceScores);
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
