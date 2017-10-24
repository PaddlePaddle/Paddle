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

#include "paddle/operators/beam_search_op.h"

#include <map>
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

bool BeamSearchAlgorithm::operator()(framework::LoDTensor *selected_ids,
                                     framework::LoDTensor *selected_scores) {
  if (framework::product(selected_ids->dims()) == 0) {
    return false;
  }

  CollectSelectedResult(selected_ids, selected_scores);
  return true;
}

void BeamSearchAlgorithm::CollectSelectedResult(
    framework::LoDTensor *selected_ids, framework::LoDTensor *selected_scores) {
  auto items = BeamSelectSourceItems();
  auto selected_items = CollectItems(items);
  // calculate tensor shape first
  size_t num_instances = std::accumulate(
      std::begin(items), std::end(items), 0,
      [](size_t a, std::vector<Item> &b) { return a + b.size(); });
  auto dims = framework::make_ddim(
      std::vector<int64_t>({static_cast<int>(num_instances), 1}));
  selected_ids->Resize(dims);
  selected_scores->Resize(dims);

  std::map<size_t /*offset*/, std::vector<Item>> hash;
  framework::LoD new_lod;
  std::vector<size_t> high_level;
  std::vector<size_t> low_level;
  size_t low_offset = 0;
  auto *ids_data = selected_ids->data<id_t>();
  auto *scores_data = selected_scores->data<score_t>();

  for (size_t high_id = 0; high_id < ids_->NumElements(lod_level_);
       high_id++) {  // source sentence level
    for (size_t low_id = 0;
         low_id < ids_->NumElements(lod_level_ + 1);  // prefix level
         low_id++) {
      size_t num_prefix = ids_->NumElements(lod_level_ + 1, low_id);
      // empty prefix, should insert an empty record into the LoD
      if (num_prefix == 0) {
        low_level.push_back(low_offset);
      } else {
        // each instance will get a candidate set.
        for (size_t i = 0; i < num_prefix; i++) {
          auto &prefix_items = selected_items[low_offset];
          for (auto &item : prefix_items) {
            ids_data[low_offset] = item.id;
            scores_data[low_offset] = item.score;
            low_offset++;
          }
          low_level.push_back(low_offset);
        }
      }
    }
  }
  // update the lowest LoD level.
  framework::LoD lod = ids_->lod();
  lod.back().clear();
  for (auto offset : low_level) {
    lod.back().push_back(offset);
  }
  selected_ids->set_lod(lod);
  selected_scores->set_lod(lod);
}

std::map<size_t, std::vector<BeamSearchAlgorithm::Item>>
BeamSearchAlgorithm::CollectItems(const std::vector<std::vector<Item>> &items) {
  std::map<size_t, std::vector<BeamSearchAlgorithm::Item>> result;
  for (size_t offset = 0; offset < items.size(); offset++) {
    for (const auto &item : items[offset]) {
      result[offset].push_back(item);
    }
  }
  return result;
}

std::vector<std::vector<BeamSearchAlgorithm::Item>>
BeamSearchAlgorithm::BeamSelectSourceItems() {
  std::vector<std::vector<Item>> result;
  std::vector<Item> items;
  while (NextSourceItems(&items)) {
    std::nth_element(std::begin(items), std::begin(items) + beam_size_,
                     std::end(items), [](const Item &a, const Item &b) {
                       // TODO(superjom) make score's comparation customizable.
                       return a.score < b.score;
                     });
    if (items.size() > beam_size_) {
      items.resize(beam_size_);
    }
    result.emplace_back(items);
  }
  return result;
}

bool BeamSearchAlgorithm::NextSourceItems(
    std::vector<BeamSearchAlgorithm::Item> *items) {
  if (seq_offset_ >= ids_->NumElements(lod_level_)) {
    return false;
  }
  // find the current candidates
  auto ids = *ids_;
  auto scores = *scores_;
  ids.ShrinkInLevel(lod_level_, seq_offset_, seq_offset_ + 1);
  scores.ShrinkInLevel(lod_level_, seq_offset_, seq_offset_ + 1);
  // transform tensor to items
  items->clear();
  items->reserve(framework::product(ids.dims()));

  PADDLE_ENFORCE_GT(ids.NumLevels(), size_t(lod_level_ + 1),
                    "more than %d+1 LoD levels should be valid", lod_level_);
  // traverse the lower-level elements as offsets
  // skip the empty set first, for that will not affect the order of sort.
  size_t instance_dim = 1;
  for (int i = 1; i < ids.dims().size(); i++) {
    instance_dim *= ids.dims()[i];
  }
  // TODO(superjom) make the type customizable.
  auto *ids_data = ids_->data<int>();
  auto *scores_data = scores_->data<float>();
  auto abs_lod = framework::ToAbsOffset(ids.lod());
  for (size_t offset = abs_lod[lod_level_].front();
       offset != abs_lod[lod_level_].back(); offset++) {
    for (size_t d = 0; d < instance_dim; d++) {
      items->emplace_back(offset, ids_data[offset + d],
                          scores_data[offset + d]);
    }
  }
  return true;
}

class BeamSearchProtoAndCheckerMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  BeamSearchProtoAndCheckerMaker(framework::OpProto *proto,
                                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    // inputs and outputs stored in proto
    AddInput("ids", "the candidate ids");
    AddInput("scores", "the scores of candidates");
    AddOutput("selected_ids", "the selected candiates");
    AddOutput("selected_scores", "the scores of the selected candidates");

    // Attributes stored in AttributeMap
    AddAttr<int>("level", "the level of LoDTensor");
    AddAttr<int>("beam_size", "beam size for beam search");

    AddComment(
        "This is a beam search operator that help to generate sequences.");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(beam_search, paddle::operators::BeamSearchOp,
                             paddle::operators::BeamSearchProtoAndCheckerMaker);
