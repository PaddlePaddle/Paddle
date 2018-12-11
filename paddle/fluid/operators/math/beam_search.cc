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

#include "paddle/fluid/operators/math/beam_search.h"
#include <algorithm>
#include <map>

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class BeamSearchFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext &context,
                  const framework::LoDTensor &pre_ids,
                  const framework::LoDTensor &pre_scores,
                  const framework::LoDTensor &ids,
                  const framework::LoDTensor &scores,
                  framework::LoDTensor *selected_ids,
                  framework::LoDTensor *selected_scores, size_t level,
                  size_t beam_size, int end_id) {
    // Input the arguments that needed by this class.
    ids_ = &ids;
    scores_ = &scores;
    beam_size_ = beam_size;
    lod_level_ = level;
    end_id_ = end_id;
    sent_offset_ = 0;

    auto abs_lod = framework::ToAbsOffset(ids_->lod());
    auto &high_level = abs_lod[lod_level_];

    LOG(INFO) << "ids.abs_lod: " << abs_lod;

    auto items = SelectTopBeamSizeItems(pre_ids, pre_scores);
    auto selected_items = ToMap(items, high_level.back());
    LOG(INFO) << "selected_items:";
    for (size_t i = 0; i < selected_items.size(); ++i) {
      LOG(INFO) << "offset:" << i;
      for (auto &item : selected_items[i]) {
        LOG(INFO) << item.ToString();
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

  /*
   * The basic items help to sort.
   */
  struct Item {
    Item() {}
    Item(size_t offset, size_t id, float score)
        : offset(offset), id(id), score(score) {}
    // offset in the higher lod level.
    size_t offset;
    // prefix id in the lower lod level.
    // size_t prefix;
    // the candidate id
    size_t id;
    // the corresponding score
    float score;

    std::string ToString() {
      std::ostringstream os;
      os << "{";
      os << "offset: " << offset << ", ";
      os << "id: " << id << ", ";
      os << "score: " << score << "";
      os << "}";
      return os.str();
    }
  };

 protected:
  /*
   * Prune the source sentences all branchs finished, and it is optional.
   * Pruning must one step later than finishing (thus pre_ids is needed here),
   * since the end tokens must be writed out.
   */
  void PruneEndBeams(const framework::LoDTensor &pre_ids,
                     std::vector<std::vector<Item>> *items) {
    auto *pre_ids_data = pre_ids.data<int64_t>();
    auto abs_lod = framework::ToAbsOffset(ids_->lod());
    auto &high_level = abs_lod[lod_level_];
    for (size_t src_idx = 0; src_idx < high_level.size() - 1; ++src_idx) {
      size_t src_prefix_start = high_level[src_idx];
      size_t src_prefix_end = high_level[src_idx + 1];
      bool finish_flag = true;
      for (size_t offset = src_prefix_start; offset < src_prefix_end;
           offset++) {
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
        for (size_t offset = src_prefix_start; offset < src_prefix_end;
             offset++)
          items->at(offset).clear();
      }
    }
  }

  /*
   * Transform the items into a map whose key is offset, value is the items.
   * NOTE low performance.
   */
  std::vector<std::vector<Item>> ToMap(
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

  /*
   * For each source, select top beam_size records.
   */
  std::vector<std::vector<Item>> SelectTopBeamSizeItems(
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
        VLOG(3) << item.ToString();
      }
    }

    return result;
  }

  /*
   * Get the items of next source sequence, return false if no remaining items.
   * the candidates of a source
   */
  bool NextItemSet(const framework::LoDTensor &pre_ids,
                   const framework::LoDTensor &pre_scores,
                   std::vector<Item> *items) {
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

 protected:
  size_t beam_size_;
  const framework::LoDTensor *ids_;
  const framework::LoDTensor *scores_;
  size_t lod_level_{0};
  size_t sent_offset_{0};
  int end_id_{0};
};

template class BeamSearchFunctor<platform::CPUDeviceContext, int>;
template class BeamSearchFunctor<platform::CPUDeviceContext, int64_t>;
template class BeamSearchFunctor<platform::CPUDeviceContext, float>;
template class BeamSearchFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
