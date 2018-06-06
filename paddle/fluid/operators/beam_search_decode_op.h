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

#pragma once

#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using LoDTensorArray = framework::LoDTensorArray;

// all the lod have 2 levels.
// The First is source level, the second is sentence level.
// source level describe how many candidate words for this source.
// sentence level describe these candidates belong to which prefix
const size_t kSourceLevel = 0;
const size_t kSentenceLevel = 1;

template <typename T>
struct BeamNode {
  BeamNode(int64_t word_id, T score) : word_id_(word_id), score_(score) {}

  ~BeamNode() {
    if (parent_) {
      parent_->DropKid(this);
      if (parent_->kids_.size() == 0UL) {
        delete parent_;
      }
    }
    VLOG(3) << "Delete BeamNode root with word_id:" << this->word_id_;
  }

  void AppendTo(BeamNode* parent) {
    parent_ = parent;
    parent->kids_.insert(this);
  }

  void DropKid(BeamNode* kid) { kids_.erase(kid); }

  BeamNode* parent_ = nullptr;
  std::unordered_set<BeamNode*> kids_;
  int64_t word_id_;
  T score_;
};

template <typename T>
using BeamNodeVector = std::vector<std::unique_ptr<BeamNode<T>>>;

template <typename T>
struct Sentence {
  std::vector<int64_t> word_ids;
  std::vector<T> scores;
};

template <typename T>
using SentenceVector = std::vector<Sentence<T>>;

template <typename T>
struct BeamSearchDecoder {
  BeamSearchDecoder(size_t beam_size, int end_id)
      : beam_size_(beam_size), end_id_(end_id) {}

  /**
   * make a BeamNode and all it's related prefix BeanNode into a Sentence.
   */
  Sentence<T> MakeSentence(const BeamNode<T>* node) const;

  /**
   * Param:
   *  cur_ids: LoDTensor of One step for word ID
   *  cur_scores: LoDTensor of One Step for word score
   *  prefixes_list: prefixes for each source sentence.
   *  sentence_vector_list: result sentence_vector for each source sentence.
   * Return:
   *  a new prefixes list for each source of current step
   */
  std::vector<BeamNodeVector<T>> PackTwoSteps(
      const LoDTensor& cur_ids, const LoDTensor& cur_scores,
      std::vector<BeamNodeVector<T>>* prefixes_list,
      std::vector<SentenceVector<T>>* sentence_vector_list) const;

  /**
   * convert the result sentence_vector for each source sentence into two
   * LodTensor.
   * One is all candidate sentences with word id, one is all candidate sentences
   * with word score.
   * Param:
   *  sentence_vector_list: sentence_vector for each source sentence.
   *  id_tensor: result LoDTensor for sentences of id.
   *  score_tensor: result LoDTensor for sentences of score.
   */
  void ConvertSentenceVectorToLodTensor(
      std::vector<SentenceVector<T>> sentence_vector_list, LoDTensor* id_tensor,
      LoDTensor* score_tensor, bool reverse = false,
      bool sort_by_score = true) const;

  /**
   * Pack all steps of id/score LodTensor into sentence LoDTensor
   * it's main logic is:
   * ```python
   *   prefix
   *   result_sentence
   *   result_lod_tensor
   *
   *   for (step in steps):
   *     prefix = PackTwoSteps(prefix, step, &result_sentence)
   *   ConvertSentenceVector<T>ToLodTensor(result_sentence, &result_lod_tensor)
   * ```
   */
  void PackAllSteps(const LoDTensorArray& step_ids,
                    const LoDTensorArray& step_scores, LoDTensor* id_tensor,
                    LoDTensor* score_tensor) const;

  void Backtrace(const LoDTensorArray& step_ids,
                 const LoDTensorArray& step_scores, LoDTensor* id_tensor,
                 LoDTensor* score_tensor) const;

  size_t beam_size_;
  int end_id_;
};

template <typename T>
Sentence<T> BeamSearchDecoder<T>::MakeSentence(const BeamNode<T>* node) const {
  Sentence<T> sentence;
  while (node != nullptr) {
    sentence.word_ids.emplace_back(node->word_id_);
    sentence.scores.emplace_back(node->score_);
    node = node->parent_;
  }

  std::reverse(std::begin(sentence.word_ids), std::end(sentence.word_ids));
  std::reverse(std::begin(sentence.scores), std::end(sentence.scores));

  return sentence;
}

template <typename T>
std::vector<BeamNodeVector<T>> BeamSearchDecoder<T>::PackTwoSteps(
    const LoDTensor& cur_ids, const LoDTensor& cur_scores,
    std::vector<BeamNodeVector<T>>* prefixes_list,
    std::vector<SentenceVector<T>>* sentence_vector_list) const {
  std::vector<BeamNodeVector<T>> result;

  for (size_t src_idx = 0; src_idx < cur_ids.lod()[kSourceLevel].size() - 1;
       ++src_idx) {
    size_t src_start = cur_ids.lod().at(kSourceLevel)[src_idx];
    size_t src_end = cur_ids.lod().at(kSourceLevel)[src_idx + 1];

    BeamNodeVector<T> beam_nodes;

    // if prefixes size is 0, it means this is the first step. In this step,
    // all candidate id is the start of candidate sentences.
    if (prefixes_list->empty()) {
      PADDLE_ENFORCE_EQ(cur_ids.lod().at(kSourceLevel).back(),
                        cur_ids.lod().at(kSentenceLevel).back(),
                        "in the first step");
      for (size_t id_idx = src_start; id_idx < src_end; ++id_idx) {
        beam_nodes.push_back(std::unique_ptr<BeamNode<T>>(new BeamNode<T>(
            cur_ids.data<int64_t>()[id_idx], cur_scores.data<T>()[id_idx])));
      }
    } else {
      BeamNodeVector<T>& prefixes = prefixes_list->at(src_idx);
      SentenceVector<T>& sentence_vector = (*sentence_vector_list)[src_idx];

      PADDLE_ENFORCE_EQ(src_end - src_start, prefixes.size(),
                        "prefix and candidate set number should be the same");

      auto candidate_offset = cur_ids.lod()[kSentenceLevel];
      for (size_t prefix_idx = 0; prefix_idx < prefixes.size(); ++prefix_idx) {
        std::unique_ptr<BeamNode<T>>& prefix = prefixes[prefix_idx];
        size_t candidate_start = candidate_offset[src_start + prefix_idx];
        size_t candidate_end = candidate_offset[src_start + prefix_idx + 1];
        if (candidate_start == candidate_end) {
          VLOG(3) << "this sentence has no more candidate, "
                     "add to result sentence and rm it from beam tree";
          sentence_vector.push_back(MakeSentence(prefix.get()));
          prefix.reset();
        } else {
          for (size_t candidate_idx = candidate_start;
               candidate_idx < candidate_end; ++candidate_idx) {
            auto* candidate =
                new BeamNode<T>(cur_ids.data<int64_t>()[candidate_idx],
                                cur_scores.data<T>()[candidate_idx]);
            candidate->AppendTo(prefix.get());
            beam_nodes.push_back(std::unique_ptr<BeamNode<T>>(candidate));
          }
          prefix.release();
        }
      }
    }
    result.push_back(std::move(beam_nodes));
  }
  return result;
}

template <typename T>
void BeamSearchDecoder<T>::ConvertSentenceVectorToLodTensor(
    std::vector<SentenceVector<T>> sentence_vector_list, LoDTensor* id_tensor,
    LoDTensor* score_tensor, bool reverse, bool sort_by_score) const {
  size_t src_num = sentence_vector_list.size();

  PADDLE_ENFORCE_NE(src_num, 0, "src_num should not be 0");

  std::vector<size_t> source_level_lod = {0};
  std::vector<size_t> sentence_level_lod = {0};
  std::vector<int64_t> id_data;
  std::vector<T> score_data;

  for (size_t src_idx = 0; src_idx < src_num; ++src_idx) {
    if (sort_by_score) {
      sort(sentence_vector_list[src_idx].begin(),
           sentence_vector_list[src_idx].end(),
           [reverse](const Sentence<T>& a, const Sentence<T>& b) {
             if (reverse)
               return a.scores.front() > b.scores.front();
             else
               return a.scores.back() > b.scores.back();
           });
    }
    for (Sentence<T>& sentence : sentence_vector_list[src_idx]) {
      if (reverse) {
        id_data.insert(id_data.end(), sentence.word_ids.rbegin(),
                       sentence.word_ids.rend());
        score_data.insert(score_data.end(), sentence.scores.rbegin(),
                          sentence.scores.rend());
      } else {
        id_data.insert(id_data.end(), sentence.word_ids.begin(),
                       sentence.word_ids.end());
        score_data.insert(score_data.end(), sentence.scores.begin(),
                          sentence.scores.end());
      }

      sentence_level_lod.push_back(sentence_level_lod.back() +
                                   sentence.word_ids.size());
    }
    source_level_lod.push_back(source_level_lod.back() +
                               sentence_vector_list[src_idx].size());
  }

  auto cpu_place = std::unique_ptr<paddle::platform::CPUPlace>(
      new paddle::platform::CPUPlace());
  paddle::platform::CPUDeviceContext cpu_ctx(*cpu_place.get());

  framework::LoD lod;
  lod.push_back(source_level_lod);
  lod.push_back(sentence_level_lod);

  id_tensor->set_lod(lod);
  id_tensor->Resize({static_cast<int64_t>(id_data.size())});
  id_tensor->mutable_data<int64_t>(paddle::platform::CPUPlace());
  framework::TensorFromVector<int64_t>(id_data, cpu_ctx, id_tensor);

  score_tensor->set_lod(lod);
  score_tensor->Resize({static_cast<int64_t>(score_data.size())});
  score_tensor->mutable_data<T>(paddle::platform::CPUPlace());
  framework::TensorFromVector<T>(score_data, cpu_ctx, score_tensor);
}

template <typename T>
void BeamSearchDecoder<T>::PackAllSteps(const LoDTensorArray& step_ids,
                                        const LoDTensorArray& step_scores,
                                        LoDTensor* id_tensor,
                                        LoDTensor* score_tensor) const {
  PADDLE_ENFORCE(!step_ids.empty(), "step num should be larger than 0");
  PADDLE_ENFORCE_EQ(step_ids.size(), step_scores.size(),
                    "step_ids and step_scores should be the same");
  const size_t step_num = step_ids.size();
  const size_t src_num = step_ids.at(0).lod().at(kSourceLevel).size() - 1;

  PADDLE_ENFORCE_GT(src_num, 0UL, "source num should be larger than 0");

  // previous prefixes for each step,
  // the init length is 0, means this is the first step.
  std::vector<BeamNodeVector<T>> beamnode_vector_list(0);
  std::vector<SentenceVector<T>> sentence_vector_list(src_num);

  // pack all steps for one batch first, then another batch
  for (size_t step_id = 0; step_id < step_num; ++step_id) {
    beamnode_vector_list =
        PackTwoSteps(step_ids.at(step_id), step_scores.at(step_id),
                     &beamnode_vector_list, &sentence_vector_list);
  }
  // append last beam_node to result
  for (size_t src_idx = 0; src_idx < src_num; ++src_idx) {
    for (auto& beam_node : beamnode_vector_list.at(src_idx)) {
      sentence_vector_list[src_idx].push_back(MakeSentence(beam_node.get()));
      beam_node.reset();
    }
  }

  ConvertSentenceVectorToLodTensor(sentence_vector_list, id_tensor,
                                   score_tensor);
}

template <typename T>
void BeamSearchDecoder<T>::Backtrace(const LoDTensorArray& step_ids,
                                     const LoDTensorArray& step_scores,
                                     LoDTensor* id_tensor,
                                     LoDTensor* score_tensor) const {
  PADDLE_ENFORCE(!step_ids.empty(), "step num should be larger than 0");
  PADDLE_ENFORCE_EQ(step_ids.size(), step_scores.size(),
                    "step_ids and step_scores should be the same");
  const size_t step_num = step_ids.size();
  const size_t src_num = step_ids.at(0).lod().at(kSourceLevel).size() - 1;
  std::vector<SentenceVector<T>> sentence_vector_list(
      src_num, SentenceVector<T>(beam_size_));
  std::vector<std::vector<size_t>> prefix_idx_vector_list(
      src_num, std::vector<size_t>());
  for (int step_id = step_num - 1; step_id >= 0; --step_id) {
    auto& cur_ids = step_ids.at(step_id);
    auto& cur_scores = step_scores.at(step_id);
    for (size_t src_idx = 0; src_idx < src_num; ++src_idx) {
      // for each source sentence
      auto& sentence_vector = sentence_vector_list.at(src_idx);
      auto& prefix_idx_vector = prefix_idx_vector_list.at(src_idx);
      size_t src_prefix_start = cur_ids.lod().at(kSourceLevel)[src_idx];
      size_t src_prefix_end = cur_ids.lod().at(kSourceLevel)[src_idx + 1];
      if (prefix_idx_vector.empty()) {  // be finished and pruned at this step
                                        // or the last time step
        for (size_t prefix_idx = src_prefix_start; prefix_idx < src_prefix_end;
             ++prefix_idx) {
          size_t candidate_start = cur_ids.lod().at(kSentenceLevel)[prefix_idx];
          size_t candidate_end =
              cur_ids.lod().at(kSentenceLevel)[prefix_idx + 1];
          for (size_t candidate_idx = candidate_start;
               candidate_idx < candidate_end; ++candidate_idx) {
            prefix_idx_vector.push_back(prefix_idx);
            size_t idx = prefix_idx_vector.size() - 1;
            auto cur_id = cur_ids.data<int64_t>()[candidate_idx];
            auto cur_score = cur_scores.data<T>()[candidate_idx];
            sentence_vector.at(idx).word_ids.push_back(cur_id);
            sentence_vector.at(idx).scores.push_back(cur_score);
          }
        }
      } else {  // use prefix_idx_vector to backtrace
        size_t src_candidate_start =
            cur_ids.lod().at(kSentenceLevel)[src_prefix_start];
        size_t prefix_idx = src_prefix_start;
        size_t candidate_num =
            cur_ids.lod().at(kSentenceLevel)[prefix_idx + 1] -
            cur_ids.lod().at(kSentenceLevel)[prefix_idx];
        for (size_t idx = 0; idx < prefix_idx_vector.size(); ++idx) {
          auto candidate_idx = prefix_idx_vector.at(idx);
          auto cur_id = cur_ids.data<int64_t>()[candidate_idx];
          auto cur_score = cur_scores.data<T>()[candidate_idx];
          if (cur_id != end_id_ || sentence_vector.at(idx).word_ids.empty()) {
            // to skip redundant end tokens
            sentence_vector.at(idx).word_ids.push_back(cur_id);
            sentence_vector.at(idx).scores.push_back(cur_score);
          }

          while (src_candidate_start + candidate_num <=
                 candidate_idx) {  // search the corresponding prefix
            prefix_idx++;
            candidate_num += cur_ids.lod().at(kSentenceLevel)[prefix_idx + 1] -
                             cur_ids.lod().at(kSentenceLevel)[prefix_idx];
          }
          prefix_idx_vector.at(idx) = prefix_idx;
        }
      }
    }
  }

  ConvertSentenceVectorToLodTensor(sentence_vector_list, id_tensor,
                                   score_tensor, true, true);
}

}  // namespace operators
}  // namespace paddle
