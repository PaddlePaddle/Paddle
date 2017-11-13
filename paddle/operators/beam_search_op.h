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

#ifdef PADDLE_WITH_TESTING
#include "gtest/gtest.h"
#endif

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

/*
 * This is an implementation of beam search.
 *
 * To explain the details, lets take machine translation task for example, in
 * this task, one source sentence is translated to multiple target sentences,
 * during this period, one sentence will be translated to multiple translation
 * prefixes(target sentence that have not ended), in each time step a prefix
 * will have some candidates, input the candidate ids and their corresponding
 * scores (probabilities), it will sort and select the top beam_size candidates
 * for each source sentence, and store the selected candidates's score and their
 * corresponding ids to LoDTensors.
 *
 * A detailed example:
 *
 * Input
 *
 * ids:
 * LoD (should have 2 levels)
 * first level: [0, 1, 3]
 * second level: [0, 1, 2, 3, 4]
 *
 * tensor's data
 * [
 * [4, 2, 5]
 * [2, 1, 3]
 * [3, 5, 2]
 * [8, 2, 1]
 * ]
 *
 * scores:
 * LoD same as `ids`
 * tensor's data
 * [
 * [0.5, 0.3, 0.2]
 * [0.6, 0.3, 0.1]
 * [0.9, 0.5, 0.1]
 * [0.7, 0.5, 0.1]
 * ]
 *
 * the inputs means that there are 2 source sentences to translate, and the
 * first source has 1 prefix, the second source has 2 prefix.
 *
 * lets assume beam size is 2, and the beam search's output should be
 * LoD
 * first level:
 * [0, 1, 2]
 * second level:
 * [0, 2, 4]
 *
 * tensor's data
 * [[
 * 0.5,
 * 0.3,
 * 0.9,
 * 0.7
 * ]]
 */
class BeamSearch {
 public:
  // TODO(superjom) make type customizable
  using id_t = size_t;
  using score_t = float;
  /*
   * Input the arguments that needed by this class.
   */
  BeamSearch(const framework::LoDTensor& ids,
             const framework::LoDTensor& scores, size_t level, size_t beam_size,
             int end_id)
      : beam_size_(beam_size),
        ids_(&ids),
        scores_(&scores),
        lod_level_(level),
        end_id_(end_id) {}

  /*
   * The main function of beam search.
   *
   * @selected_ids: a [None, 1]-shaped tensor with LoD.
   *   In a machine translation model, it might be the candidate term id sets,
   *   each set stored as a varience-length sequence.
   *   The format might be described with a two-level LoD
   *   - [[0 1]
   *   -  [0 1 2]]
   *   - [[]
   *   -  [0 1]]
   *   the first level of LoD tells that there are two source sentences. The
   *   second level describes the details of the candidate id set's offsets in
   * the
   *   source sentences.
   *
   *  @selected_scores: a LoD tensor with the same shape and LoD with
   * selected_ids.
   *   It stores the corresponding scores of candidate ids in selected_ids.
   *
   * Return false if all the input tensor is empty, in machine translation task
   * that means no candidates is provided, and the task will stop running.
   */
  void operator()(const framework::LoDTensor& pre_ids,
                  framework::LoDTensor* selected_ids,
                  framework::LoDTensor* selected_scores);

 protected:
  /*
   * The basic items help to sort.
   */
  struct Item {
    Item() {}
    Item(size_t offset, size_t id, float score)
        : offset(offset), id(id), score(score) {}
    // offset in the lod_level_+1
    size_t offset;
    // the candidate id
    id_t id;
    // the corresponding score
    score_t score;
  };

  void PruneEndidCandidates(const framework::LoDTensor& pre_ids,
                            std::vector<std::vector<Item>>* items);

  /*
   * Transform the items into a map whose key is offset, value is the items.
   * NOTE low performance
   */
  std::vector<std::vector<Item>> ToMap(
      const std::vector<std::vector<Item>>& inputs);

  /*
   * For each source, select top beam_size records.
   */
  std::vector<std::vector<Item>> SelectTopBeamSizeItems();

  /*
   * Get the items of next source sequence, return false if no remaining items.
   */
  bool NextItemSet(std::vector<Item>* items);

 private:
  size_t beam_size_;
  const framework::LoDTensor* ids_;
  const framework::LoDTensor* scores_;
  size_t lod_level_{0};
  size_t sent_offset_{0};
  int end_id_{0};
};

class BeamSearchOp : public framework::OperatorBase {
 public:
  BeamSearchOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  BeamSearchOp(const BeamSearchOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    PADDLE_THROW("Not Implemented");
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    LOG(INFO) << "run beam search op";
    auto ids_var = scope.FindVar(Input("ids"));
    auto scores_var = scope.FindVar(Input("scores"));
    auto pre_ids_var = scope.FindVar(Input("pre_ids"));
    PADDLE_ENFORCE_NOT_NULL(ids_var);
    PADDLE_ENFORCE_NOT_NULL(scores_var);
    PADDLE_ENFORCE_NOT_NULL(pre_ids_var);

    auto& ids = ids_var->Get<framework::LoDTensor>();
    auto& scores = scores_var->Get<framework::LoDTensor>();
    auto& pre_ids = pre_ids_var->Get<framework::LoDTensor>();
    size_t level = Attr<int>("level");
    size_t beam_size = Attr<int>("beam_size");
    int end_id = Attr<int>("end_id");
    LOG(INFO) << "init beam search";
    BeamSearch alg(ids, scores, level, beam_size, end_id);

    LOG(INFO) << "after beam search";
    auto selected_ids_var = scope.FindVar(Output("selected_ids"));
    auto selected_scores_var = scope.FindVar(Output("selected_scores"));
    PADDLE_ENFORCE_NOT_NULL(selected_ids_var);
    PADDLE_ENFORCE_NOT_NULL(selected_scores_var);
    auto& selected_ids_tensor =
        *selected_ids_var->GetMutable<framework::LoDTensor>();
    auto& selected_scores_tensor =
        *selected_scores_var->GetMutable<framework::LoDTensor>();
    LOG(INFO) << "run beam search";
    alg(pre_ids, &selected_ids_tensor, &selected_scores_tensor);
    LOG(INFO) << "finish beam search";
  }
};

}  // namespace operators
}  // namespace paddle
