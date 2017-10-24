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

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

/*
 * This is an implementation of beam search algorithm, input the candidate ids
 * and candidate scores, it will sort and select the top beam_size candidates,
 * return the corresponding ids and scores.
 */
class BeamSearchAlgorithm {
 public:
  // TODO(superjom) make type customizable
  using id_t = size_t;
  using score_t = float;
  /*
   * Input the arguments that needed by this class.
   */
  BeamSearchAlgorithm(const framework::LoDTensor& ids,
                      const framework::LoDTensor& scores, size_t level,
                      size_t beam_size)
      : beam_size_(beam_size),
        ids_(&ids),
        scores_(&scores),
        lod_level_(level) {}

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
  bool operator()(framework::LoDTensor* selected_ids,
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

  /*
   * Collect the candidate ids that is selected by beam search and store them
   * into a LoDTensor.
   */
  void CollectSelectedResult(framework::LoDTensor* selected_ids,
                             framework::LoDTensor* selected_scores);

  /*
   * Transform the items into a map whose key is offset, value is the items.
   * NOTE low performance
   */
  std::map<size_t, std::vector<Item>> CollectItems(
      const std::vector<std::vector<Item>>& inputs);

  /*
   * For each source, select top beam_size records.
   */
  std::vector<std::vector<Item>> BeamSelectSourceItems();

  /*
   * Get the items of next source sequence, return false if no remaining items.
   */
  bool NextSourceItems(std::vector<Item>* items);

 private:
  size_t beam_size_;
  const framework::LoDTensor* ids_;
  const framework::LoDTensor* scores_;
  size_t lod_level_{0};
  size_t seq_offset_{0};
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
    auto ids_var = scope.FindVar(Input("ids"));
    auto scores_var = scope.FindVar(Input("scores"));
    PADDLE_ENFORCE_NOT_NULL(ids_var);
    PADDLE_ENFORCE_NOT_NULL(scores_var);

    auto& ids = ids_var->Get<framework::LoDTensor>();
    auto& scores = scores_var->Get<framework::LoDTensor>();
    size_t level = Attr<int>("level");
    size_t beam_size = Attr<int>("beam_size");
    BeamSearchAlgorithm alg(ids, scores, level, beam_size);
  }
};

}  // namespace operators
}  // namespace paddle
