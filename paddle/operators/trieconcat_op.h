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

const int64_t kInitLength = 1024;
const int64_t kEndId = 0;

struct BeamNode {
  BeamNode(int64_t word_id, float prob) : word_id_(word_id), prob_(prob) {}

  void AppendTo(BeamNode* father) {
    father_ = father;
    father->kids_.push_back(this);
  }

  BeamNode* father_ = nullptr;
  std::vector<BeamNode*> kids_;
  int64_t word_id_;
  float prob_;
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

  template <typename T>
  void AppendVector(const std::vector<T> vec, LoDTensor* dst) {
    std::vector<size_t> sentences = dst->lod().at(1);
    T* tensor_data = dst->data<T>() + sentences.back();
    memcpy(tensor_data, vec.data(), vec.size() * sizeof(T));
    dst->mutable_lod()->at(1).push_back(sentences.back() + vec.size());
  }

  void AppendBeamNodeToLoDTensor(BeamNode* node, LoDTensor* dst_ids,
                                 LoDTensor* dst_probs) {
    std::vector<int64_t> sequence_ids;
    std::vector<float> sequence_probs;
    BeamNode* tmp = node;
    while (tmp != nullptr) {
      sequence_ids.push_back(tmp->word_id_);
      sequence_probs.push_back(tmp->prob_);
      tmp = tmp->father_;
    }

    std::reverse(std::begin(sequence_ids), std::end(sequence_ids));
    std::reverse(std::begin(sequence_probs), std::end(sequence_probs));

    AppendVector<int64_t>(sequence_ids, dst_ids);
    AppendVector<float>(sequence_probs, dst_probs);
  }

  std::vector<BeamNode*> PackTwoBeamStepOut(
      size_t batch_start, const std::vector<BeamNode*>& pre_results,
      const LoDTensor& cur_ids, const LoDTensor& cur_probs,
      LoDTensor* result_seq_ids, LoDTensor* result_probs) {
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
          AppendBeamNodeToLoDTensor(prefix_end, result_seq_ids, result_probs);
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

}  // namespace operators
}  // namespace paddle
