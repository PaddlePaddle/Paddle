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

#pragma once

#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using value_type = float;
using LoDTensor = framework::LoDTensor;

const int64_t kEndId = 0;

// all the lod have 2 level, the first it source level,
// each source have multiple possible sentences in the second level
const size_t kSourceLevel = 0;
const size_t kSentenceLevel = 1;

struct BeamNode {
  BeamNode(int64_t word_id, float score) : word_id_(word_id), score_(score) {}

  void AppendTo(BeamNode* father) {
    father_ = father;
    father->kids_.push_back(this);
  }

  BeamNode* father_ = nullptr;
  std::vector<BeamNode*> kids_;
  int64_t word_id_;
  float score_;
};

struct BeamHelpter {
  // remove this prefix from the beam tree
  void RemoveFromEnd(BeamNode* end) {
    PADDLE_ENFORCE_EQ(end->kids_.size(), 0UL, "end should not have any kids");
    auto* father = end->father_;
    if (father != nullptr) {
      // should use reference
      auto& kids = father->kids_;
      kids.erase(std::remove(kids.begin(), kids.end(), end), kids.end());
      VLOG(3) << "Delete BeamNode with word_id:" << end->word_id_;
      delete end;
      if (father->kids_.size() == 0) {
        RemoveFromEnd(father);
      }
    } else {
      VLOG(3) << "Delete BeamNode root with word_id:" << end->word_id_;
      delete end;
    }
  }

  void AppendBeamNodeToResult(
      size_t source_idx, const BeamNode* node,
      std::unordered_map<size_t, std::vector<std::vector<int64_t>>>* result_id,
      std::unordered_map<size_t, std::vector<std::vector<float>>>*
          result_score) {
    std::vector<int64_t> sequence_ids;
    std::vector<float> sequence_scores;

    const BeamNode* tmp = node;
    while (tmp != nullptr) {
      sequence_ids.emplace_back(tmp->word_id_);
      sequence_scores.emplace_back(tmp->score_);
      tmp = tmp->father_;
    }

    std::reverse(std::begin(sequence_ids), std::end(sequence_ids));
    std::reverse(std::begin(sequence_scores), std::end(sequence_scores));

    (*result_id)[source_idx].emplace_back(sequence_ids);
    (*result_score)[source_idx].push_back(sequence_scores);
  }

  std::vector<BeamNode*> PackTwoBeamStepOut(
      size_t source_idx, const std::vector<BeamNode*>& prefixes,
      const LoDTensor& cur_ids, const LoDTensor& cur_scores,
      std::unordered_map<size_t, std::vector<std::vector<int64_t>>>* result_id,
      std::unordered_map<size_t, std::vector<std::vector<float>>>*
          result_score) {
    std::vector<BeamNode*> result;

    size_t source_start = cur_ids.lod()[kSourceLevel][source_idx];
    size_t source_end = cur_ids.lod()[kSourceLevel][source_idx + 1];
    PADDLE_ENFORCE_EQ(source_end - source_start, prefixes.size(),
                      "prefix and candidate set number should be the same");
    std::vector<size_t> candidate_offset = cur_ids.lod()[kSentenceLevel];
    for (size_t prefix_idx = 0; prefix_idx < prefixes.size(); ++prefix_idx) {
      size_t candidate_start = candidate_offset[source_start + prefix_idx];
      size_t candidate_end = candidate_offset[source_start + prefix_idx + 1];
      auto* prefix = prefixes[prefix_idx];
      PADDLE_ENFORCE_NE(prefix->word_id_, kEndId,
                        "prefix should not contain end id");
      if (candidate_start == candidate_end) {
        VLOG(3) << "this sentence has no more candidate, prune it";
        // remove this sentence from Beam Tree.
        RemoveFromEnd(prefix);
      } else {
        // two level lod
        // [0 2 6] source level
        // [0 1 1 2 3 4] sentence level
        PADDLE_ENFORCE_NE(prefix->word_id_, kEndId,
                          "End id should not have candidate anymore");
        for (size_t candidate_index = candidate_start;
             candidate_index < candidate_end; ++candidate_index) {
          int64_t word_id = cur_ids.data<int64_t>()[candidate_index];
          float score = cur_scores.data<float>()[candidate_index];
          auto* candidate = new BeamNode(word_id, score);
          candidate->AppendTo(prefix);
          // if candidate is end id, then put it into result and remove it from
          // beam tree.
          if (word_id == kEndId) {
            AppendBeamNodeToResult(source_idx, candidate, result_id,
                                   result_score);
            RemoveFromEnd(candidate);
          } else {
            result.push_back(candidate);
          }
        }
      }
    }
    return result;
  }

  void InitFirstStepBeamNodes(
      const LoDTensor& tensor_id, const LoDTensor& tensor_score,
      std::unordered_map<size_t, std::vector<BeamNode*>>* batch_beam_nodes) {
    // init beam_nodes for each source sentence.
    // in the first level, each sentence should have be a prefix
    // [0 3 6] level 0
    // [0 1 2 3 4 5 6] level 1
    // [0 0 0 0 0 0] data
    PADDLE_ENFORCE_EQ(tensor_id.lod().at(kSourceLevel).back(),
                      tensor_id.lod().at(kSentenceLevel).back());

    const size_t source_num = tensor_id.lod().at(kSourceLevel).size() - 1;

    for (size_t source_idx = 0; source_idx < source_num; ++source_idx) {
      std::vector<BeamNode*> init_beam_nodes;
      size_t source_start = tensor_id.lod().at(kSourceLevel).at(source_idx);
      size_t source_end = tensor_id.lod().at(kSourceLevel).at(source_idx + 1);

      for (size_t word_id_idx = source_start; word_id_idx < source_end;
           ++word_id_idx) {
        init_beam_nodes.push_back(
            new BeamNode(tensor_id.data<int64_t>()[word_id_idx],
                         tensor_score.data<float>()[word_id_idx]));
      }
      (*batch_beam_nodes)[source_idx] = init_beam_nodes;
    }
  }

  void ConvertMapToLodTensor(
      const std::unordered_map<size_t, std::vector<std::vector<int64_t>>>&
          result_id,
      const std::unordered_map<size_t, std::vector<std::vector<float>>>&
          result_score,
      LoDTensor* id_tensor, LoDTensor* score_tensor) const {
    size_t source_num = result_id.size();

    std::vector<size_t> source_level_lod = {0};
    std::vector<size_t> sentence_level_lod = {0};
    std::vector<int64_t> id_data;
    std::vector<float> score_data;
    for (size_t source_idx = 0; source_idx < source_num; ++source_idx) {
      auto& all_sentence_ids = result_id.at(source_idx);
      auto& all_sentence_scores = result_score.at(source_idx);
      for (size_t sentence_idx = 0; sentence_idx < all_sentence_ids.size();
           ++sentence_idx) {
        auto& sentence_ids = all_sentence_ids.at(sentence_idx);
        id_data.insert(id_data.end(), sentence_ids.begin(), sentence_ids.end());
        auto& sentence_scores = all_sentence_scores.at(sentence_idx);
        score_data.insert(score_data.end(), sentence_scores.begin(),
                          sentence_scores.end());
        sentence_level_lod.push_back(sentence_level_lod.back() +
                                     sentence_ids.size());
      }
      source_level_lod.push_back(source_level_lod.back() +
                                 all_sentence_ids.size());
    }

    auto cpu_place = new paddle::platform::CPUPlace();
    paddle::platform::CPUDeviceContext cpu_ctx(*cpu_place);

    framework::LoD lod;
    lod.push_back(source_level_lod);
    lod.push_back(sentence_level_lod);

    id_tensor->set_lod(lod);
    id_tensor->Resize({static_cast<int64_t>(id_data.size())});
    id_tensor->mutable_data<int64_t>(paddle::platform::CPUPlace());
    id_tensor->CopyFromVector<int64_t>(id_data, cpu_ctx);

    score_tensor->set_lod(lod);
    score_tensor->Resize({static_cast<int64_t>(score_data.size())});
    score_tensor->mutable_data<float>(paddle::platform::CPUPlace());
    score_tensor->CopyFromVector<float>(score_data, cpu_ctx);
  }

  void PackAllSteps(const std::vector<LoDTensor>& step_ids,
                    const std::vector<LoDTensor>& step_scores,
                    LoDTensor* id_tensor, LoDTensor* score_tensor) {
    PADDLE_ENFORCE_EQ(step_ids.size(), step_scores.size(),
                      "step_ids and step_scores should be the same");
    size_t step_num = step_ids.size();
    size_t source_num = step_ids.at(0).lod().at(kSourceLevel).size() - 1;

    std::unordered_map<size_t, std::vector<BeamNode*>> batch_beam_nodes;
    std::unordered_map<size_t, std::vector<std::vector<int64_t>>> result_id;
    std::unordered_map<size_t, std::vector<std::vector<float>>> result_score;

    InitFirstStepBeamNodes(step_ids.at(0), step_scores.at(0),
                           &batch_beam_nodes);

    // pack all steps for one batch first, then another batch
    for (size_t source_idx = 0; source_idx < source_num; ++source_idx) {
      for (size_t step_id = 1; step_id < step_num; ++step_id) {
        auto prefixes = batch_beam_nodes.at(source_idx);
        if (prefixes.size() > 0UL) {
          std::vector<BeamNode*> result = PackTwoBeamStepOut(
              source_idx, prefixes, step_ids.at(step_id),
              step_scores.at(step_id), &result_id, &result_score);
          batch_beam_nodes[source_idx] = result;
        } else {
          VLOG(3) << "source_idx: " << source_idx << " step_id: " << step_id
                  << " have no more candidate";
        }
      }

      // append last beam_node to result
      for (auto* beam_node : batch_beam_nodes.at(source_idx)) {
        AppendBeamNodeToResult(source_idx, beam_node, &result_id,
                               &result_score);
        RemoveFromEnd(beam_node);
      }
    }

    ConvertMapToLodTensor(result_id, result_score, id_tensor, score_tensor);
  }
};

}  // namespace operators
}  // namespace paddle
