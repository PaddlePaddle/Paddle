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
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

bool BeamSearch::operator()(framework::LoDTensor *selected_ids,
                            framework::LoDTensor *selected_scores) {
  ToLoDTensor(selected_ids, selected_scores);
  return true;
}

void BeamSearch::ToLoDTensor(framework::LoDTensor *selected_ids,
                             framework::LoDTensor *selected_scores) {
  auto items = SelectTopBeamSizeItems();
  auto selected_items = ToMap(items);
  // calculate the output tensor's height
  size_t num_instances = std::accumulate(
      std::begin(items), std::end(items), 0,
      [](size_t a, std::vector<Item> &b) { return a + b.size(); });
  // the output tensor shape should be [num_instances, 1]
  auto dims = framework::make_ddim(
      std::vector<int64_t>({static_cast<int>(num_instances), 1}));
  selected_ids->Resize(dims);
  selected_scores->Resize(dims);

  std::map<size_t /*offset*/, std::vector<Item>> hash;
  framework::LoD new_lod;
  auto *ids_data = selected_ids->mutable_data<int>(platform::CPUPlace());
  auto *scores_data =
      selected_scores->mutable_data<float>(platform::CPUPlace());

  // fill in data
  std::vector<size_t> low_level;
  size_t low_offset = 0;
  for (auto &items : selected_items) {
    low_level.push_back(low_offset);
    for (auto &item : items) {
      ids_data[low_offset] = item.id;
      scores_data[low_offset] = item.score;
      low_offset++;
    }
  }
  // fill lod
  auto abs_lod = framework::ToAbsOffset(ids_->lod());
  auto& high_level = abs_lod[lod_level_];
  framework::LoD lod(2);
  lod[0].assign(high_level.begin(), high_level.end());
  lod[1].assign(low_level.begin(), low_level.end());
  selected_ids->set_lod(lod);
  selected_scores->set_lod(lod);
}

std::vector<std::vector<BeamSearch::Item>> BeamSearch::ToMap(
    const std::vector<std::vector<Item>> &items) {
  std::vector<std::vector<Item>> result;
  for (auto &entries : items) {
    for (const auto &item : entries) {
      if (item.offset >= result.size()) {
        result.resize(item.offset + 1);
      }
      result[item.offset].push_back(item);
    }
  }
  return result;
}

std::vector<std::vector<BeamSearch::Item>>
BeamSearch::SelectTopBeamSizeItems() {
  std::vector<std::vector<Item>> result;
  std::vector<Item> items;
  // for each source sentence, select the top beam_size items across all
  // candidate sets.
  while (NextItemSet(&items)) {
    std::nth_element(std::begin(items), std::begin(items) + beam_size_,
                     std::end(items), [](const Item &a, const Item &b) {
                       // TODO(superjom) make score's comparation customizable.
                       // partial sort in descending order
                       return a.score > b.score;
                     });
    // prune the top beam_size items.
    if (items.size() > beam_size_) {
      items.resize(beam_size_);
    }
    result.emplace_back(items);
  }
  return result;
}

// the candidates of a source
bool BeamSearch::NextItemSet(std::vector<BeamSearch::Item> *items) {
  if (sent_offset_ >= ids_->NumElements(lod_level_)) {
    return false;
  }
  // find the current candidates
  auto ids = *ids_;
  auto scores = *scores_;

  auto source_abs_two_level_lod = framework::SliceInLevel(
      ids.lod(), lod_level_, sent_offset_, sent_offset_ + 1);
  source_abs_two_level_lod = framework::ToAbsOffset(source_abs_two_level_lod);
  auto abs_lod = framework::ToAbsOffset(ids.lod());
  PADDLE_ENFORCE_GE(source_abs_two_level_lod.size(), 2UL);

  auto *ids_data = ids.data<int>();
  auto *scores_data = scores.data<float>();

  size_t instance_dim = 1;
  for (int i = 1; i < ids.dims().size(); i++) {
    instance_dim *= ids.dims()[i];
  }

  items->clear();
  items->reserve(framework::product(ids.dims()));
  for (size_t offset = abs_lod[lod_level_][sent_offset_];
       offset < abs_lod[lod_level_][sent_offset_ + 1]; offset++) {
    for (int d = 0; d < instance_dim; d++) {
      const size_t dim_offset = offset * instance_dim + d;
      items->emplace_back(offset, ids_data[dim_offset],
                          scores_data[dim_offset]);
    }
  }

  sent_offset_++;
  return true;
}

class BeamSearchProtoAndCheckerMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  BeamSearchProtoAndCheckerMaker(framework::OpProto *proto,
                                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    // inputs and outputs stored in proto
    AddInput("ids", "a LoDTensor of shape of [None,k]");
    AddInput("scores",
             "a LoDTensor that has the same shape and LoD with `ids`");
    AddOutput("selected_ids",
              "a LoDTensor that stores the IDs selected by beam search");
    AddOutput(
        "selected_scores",
        "a LoDTensor that has the same shape and LoD with `selected_ids`");

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
