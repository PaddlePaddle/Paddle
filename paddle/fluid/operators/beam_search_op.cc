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

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/beam_search_op.h"

namespace paddle {
namespace operators {

void BeamSearch::operator()(const framework::LoDTensor &pre_ids,
                            const framework::LoDTensor &pre_scores,
                            framework::LoDTensor *selected_ids,
                            framework::LoDTensor *selected_scores) {
  auto abs_lod = framework::ToAbsOffset(ids_->lod());
  auto &high_level = abs_lod[lod_level_];

  auto items = SelectTopBeamSizeItems(pre_ids, pre_scores);
  auto selected_items = ToMap(items, high_level.back());
  VLOG(3) << "selected_items:";
  for (size_t i = 0; i < selected_items.size(); ++i) {
    VLOG(3) << "offset:" << i;
    for (auto &item : selected_items[i]) {
      VLOG(3) << ItemToString(item);
    }
  }

  PruneEndBeams(pre_ids, &selected_items);
  // calculate the output tensor's height
  size_t num_instances = std::accumulate(
      std::begin(selected_items), std::end(selected_items), 0,
      [](size_t a, std::vector<Item> &b) { return a + b.size(); });
  // the output tensor shape should be [num_instances, 1]
  auto dims = framework::make_ddim(
      std::vector<int64_t>({static_cast<int>(num_instances), 1}));
  selected_ids->Resize(dims);
  selected_scores->Resize(dims);

  std::map<size_t /*offset*/, std::vector<Item>> hash;
  framework::LoD new_lod;
  auto *ids_data = selected_ids->mutable_data<int64_t>(platform::CPUPlace());
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
  low_level.push_back(low_offset);

  // fill lod
  framework::LoD lod(2);
  lod[0].assign(high_level.begin(), high_level.end());
  lod[1].assign(low_level.begin(), low_level.end());
  if (!framework::CheckLoD(lod)) {
    PADDLE_THROW("lod %s is not right", framework::LoDToString(lod));
  }
  selected_ids->set_lod(lod);
  selected_scores->set_lod(lod);
}

void BeamSearch::PruneEndBeams(const framework::LoDTensor &pre_ids,
                               std::vector<std::vector<Item>> *items) {
  auto *pre_ids_data = pre_ids.data<int64_t>();
  auto abs_lod = framework::ToAbsOffset(ids_->lod());
  auto &high_level = abs_lod[lod_level_];
  for (size_t src_idx = 0; src_idx < high_level.size() - 1; ++src_idx) {
    size_t src_prefix_start = high_level[src_idx];
    size_t src_prefix_end = high_level[src_idx + 1];
    bool finish_flag = true;
    for (size_t offset = src_prefix_start; offset < src_prefix_end; offset++) {
      for (auto &item : items->at(offset)) {
        if (item.id != static_cast<size_t>(end_id_) ||
            pre_ids_data[offset] != end_id_) {
          finish_flag = false;
          break;
        }
      }
      if (!finish_flag) break;
    }
    if (finish_flag) {  // all branchs of the beam (source sentence) end and
                        // prune this beam
      for (size_t offset = src_prefix_start; offset < src_prefix_end; offset++)
        items->at(offset).clear();
    }
  }
}

std::vector<std::vector<BeamSearch::Item>> BeamSearch::ToMap(
    const std::vector<std::vector<Item>> &items, size_t element_num) {
  std::vector<std::vector<Item>> result;
  result.resize(element_num);
  for (auto &entries : items) {
    for (const auto &item : entries) {
      result[item.offset].push_back(item);
    }
  }
  return result;
}

std::vector<std::vector<BeamSearch::Item>> BeamSearch::SelectTopBeamSizeItems(
    const framework::LoDTensor &pre_ids,
    const framework::LoDTensor &pre_scores) {
  std::vector<std::vector<Item>> result;
  std::vector<Item> items;
  // for each source sentence, select the top beam_size items across all
  // candidate sets.
  while (NextItemSet(pre_ids, pre_scores, &items)) {
    std::nth_element(
        std::begin(items), std::begin(items) + beam_size_, std::end(items),
        [](const Item &a, const Item &b) { return a.score > b.score; });
    // prune the top beam_size items.
    if (items.size() > beam_size_) {
      items.resize(beam_size_);
    }
    result.emplace_back(items);
  }
  VLOG(3) << "SelectTopBeamSizeItems result size " << result.size();
  for (auto &items : result) {
    VLOG(3) << "item set:";
    for (auto &item : items) {
      VLOG(3) << ItemToString(item);
    }
  }

  return result;
}

// the candidates of a source
bool BeamSearch::NextItemSet(const framework::LoDTensor &pre_ids,
                             const framework::LoDTensor &pre_scores,
                             std::vector<BeamSearch::Item> *items) {
  if (sent_offset_ >= ids_->NumElements(lod_level_)) {
    return false;
  }
  // find the current candidates
  auto ids = *ids_;
  auto scores = *scores_;

  auto abs_lod = framework::ToAbsOffset(ids.lod());

  auto *ids_data = ids.data<int64_t>();
  auto *scores_data = scores.data<float>();

  size_t instance_dim = 1;
  for (int i = 1; i < ids.dims().size(); i++) {
    instance_dim *= ids.dims()[i];
  }

  auto *pre_ids_data = pre_ids.data<int64_t>();
  auto *pre_scores_data = pre_scores.data<float>();
  items->clear();
  items->reserve(framework::product(ids.dims()));
  for (size_t offset = abs_lod[lod_level_][sent_offset_];
       offset < abs_lod[lod_level_][sent_offset_ + 1]; offset++) {
    auto pre_id = pre_ids_data[offset];
    auto pre_score = pre_scores_data[offset];
    if (pre_id == end_id_) {
      // Allocate all probability mass to eos_id for finished branchs and the
      // other candidate ids can be ignored.
      items->emplace_back(offset, end_id_, pre_score);
    } else {
      for (size_t d = 0; d < instance_dim; d++) {
        const size_t dim_offset = offset * instance_dim + d;
        items->emplace_back(offset, ids_data[dim_offset],
                            scores_data[dim_offset]);
      }
    }
  }

  sent_offset_++;
  return true;
}

std::ostream &operator<<(std::ostream &os, const BeamSearch::Item &item) {
  os << "{";
  os << "offset: " << item.offset << ", ";
  os << "id: " << item.id << ", ";
  os << "score: " << item.score << "";
  os << "}";

  return os;
}

std::string ItemToString(const BeamSearch::Item &item) {
  std::ostringstream stream;
  stream << item;
  return stream.str();
}

class BeamSearchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // inputs and outputs stored in proto
    AddInput("pre_ids",
             "(LoDTensor) The LoDTensor containing the selected ids at the "
             "previous step. It should be a tensor with shape (batch_size, 1) "
             "and lod `[[0, 1, ... , batch_size], [0, 1, ..., batch_size]]` at "
             "thefirst step.");
    AddInput("pre_scores",
             "(LoDTensor) The LoDTensor containing the accumulated "
             "scores corresponding to the selected ids at the previous step.");
    AddInput("ids",
             "(LoDTensor) The LoDTensor containing the candidates ids. Its "
             "shape should be (batch_size * beam_size, K), where K supposed to "
             "be beam_size.");
    AddInput("scores",
             "(LoDTensor) The LodTensor containing the accumulated scores "
             "corresponding to Input(ids) and its shape is the same as the "
             "shape of Input(ids).");
    AddOutput("selected_ids",
              "A LodTensor that stores the IDs selected by beam search.");
    AddOutput("selected_scores",
              "A LoDTensor containing the accumulated scores corresponding to "
              "Output(selected_ids).");

    // Attributes stored in AttributeMap
    AddAttr<int>("level", "the level of LoDTensor");
    AddAttr<int>("beam_size", "beam size for beam search");
    AddAttr<int>("end_id",
                 "the token id which indicates the end of a sequence");

    AddComment(R"DOC(
This operator does the search in beams for one time step. 
Specifically, it selects the top-K candidate word ids of current step from
Input(ids) according to their Input(scores) for all source sentences,
where K is Attr(beam_size) and Input(ids), Input(scores) are predicted results
from the computation cell. Additionally, Input(pre_ids) and Input(pre_scores)
are the output of beam_search at previous step, they are needed for special use
to handle ended candidate translations. The paths linking prefixes and selected
candidates are organized and reserved in lod.

Note that the Input(scores) passed in should be accumulated scores, and
length penalty should be done with extra operators before calculating the
accumulated scores if needed, also suggest finding top-K before it and
using the top-K candidates following.
)DOC");
  }
};

class BeamSearchOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    for (const std::string &arg :
         std::vector<std::string>({"pre_ids", "ids", "scores"})) {
      PADDLE_ENFORCE(ctx->HasInput(arg), "BeamSearch need input argument '%s'",
                     arg);
    }
    for (const std::string &arg :
         std::vector<std::string>({"selected_ids", "selected_scores"})) {
      PADDLE_ENFORCE(ctx->HasOutput(arg),
                     "BeamSearch need output argument '%s'", arg);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = framework::OpKernelType(
        framework::ToDataType(
            ctx.Input<framework::LoDTensor>("pre_ids")->type()),
        platform::CPUPlace());
    return kt;
  }
};

class BeamSearchInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    for (auto &o : op_desc.Output("selected_ids")) {
      auto &selected_ids = block->FindRecursiveOrCreateVar(o);
      selected_ids.SetType(framework::proto::VarType::LOD_TENSOR);
    }
    for (auto &o : op_desc.Output("selected_scores")) {
      auto &selected_scores = block->FindRecursiveOrCreateVar(o);
      selected_scores.SetType(framework::proto::VarType::LOD_TENSOR);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(beam_search, ops::BeamSearchOp, ops::BeamSearchOpMaker,
                  ops::BeamSearchInferVarType);
REGISTER_OP_CPU_KERNEL(
    beam_search,
    ops::BeamSearchOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BeamSearchOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::BeamSearchOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::BeamSearchOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
