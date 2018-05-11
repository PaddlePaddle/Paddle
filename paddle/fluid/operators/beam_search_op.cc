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

#include "paddle/fluid/operators/beam_search_op.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

void BeamSearch::operator()(const framework::LoDTensor &pre_ids,
                            framework::LoDTensor *selected_ids,
                            framework::LoDTensor *selected_scores) {
  auto abs_lod = framework::ToAbsOffset(ids_->lod());
  auto &high_level = abs_lod[lod_level_];

  auto items = SelectTopBeamSizeItems();
  auto selected_items = ToMap(items, high_level.back());
  VLOG(3) << "selected_items:";
  for (size_t i = 0; i < selected_items.size(); ++i) {
    VLOG(3) << "offset:" << i;
    for (auto &item : selected_items[i]) {
      VLOG(3) << ItemToString(item);
    }
  }
  PruneEndidCandidates(pre_ids, &selected_items);
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
    sort(items.begin(), items.end(), [](const Item &a, const Item &b) {
      if (a.offset < b.offset) {
        return true;
      }
      return a.id < b.id;
    });
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

int BeamSearch::PruneEndidCandidates(const framework::LoDTensor &pre_ids,
                                     std::vector<std::vector<Item>> *items) {
  auto *pre_ids_data = pre_ids.data<int64_t>();

  int res = 0;
  for (size_t offset = 0; offset < items->size(); offset++) {
    auto prefix_id = pre_ids_data[offset];
    if (prefix_id == end_id_) {
      items->at(offset).clear();
    } else {
      res++;
    }
  }

  return res;
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
bool BeamSearch::NextItemSet(std::vector<BeamSearch::Item> *items) {
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

  items->clear();
  items->reserve(framework::product(ids.dims()));
  for (size_t offset = abs_lod[lod_level_][sent_offset_];
       offset < abs_lod[lod_level_][sent_offset_ + 1]; offset++) {
    for (size_t d = 0; d < instance_dim; d++) {
      const size_t dim_offset = offset * instance_dim + d;
      items->emplace_back(offset, ids_data[dim_offset],
                          scores_data[dim_offset]);
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
    AddInput("pre_ids", "ids in previous step");
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
    AddAttr<int>("end_id",
                 "the token id which indicates the end of a sequence");

    AddComment(
        "This is a beam search operator that help to generate sequences.");
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
      block->Var(o)->SetType(framework::proto::VarType::LOD_TENSOR);
    }
    for (auto &o : op_desc.Output("selected_scores")) {
      block->Var(o)->SetType(framework::proto::VarType::LOD_TENSOR);
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
