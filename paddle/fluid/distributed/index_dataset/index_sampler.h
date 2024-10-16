// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <vector>

#include "paddle/fluid/distributed/index_dataset/index_wrapper.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/kernels/funcs/math/sampler.h"

namespace paddle {
namespace distributed {

class IndexSampler {
 public:
  virtual ~IndexSampler() {}
  IndexSampler() {}

  template <typename T>
  static std::shared_ptr<IndexSampler> Init(const std::string& name) {
    std::shared_ptr<IndexSampler> instance = nullptr;
    instance.reset(new T(name));
    return instance;
  }

  virtual void init_layerwise_conf(
      const std::vector<uint16_t>& layer_sample_counts UNUSED,
      uint16_t start_sample_layer UNUSED = 1,
      uint16_t seed UNUSED = 0) {}
  virtual void init_beamsearch_conf(const int64_t k UNUSED) {}
  virtual std::vector<std::vector<uint64_t>> sample(
      const std::vector<std::vector<uint64_t>>& user_inputs,
      const std::vector<uint64_t>& input_targets,
      bool with_hierarchy = false) = 0;

  virtual void sample_from_dataset(
      const uint16_t sample_slot,
      std::vector<paddle::framework::Record>* src_datas,
      std::vector<paddle::framework::Record>* sample_results) = 0;
};

class LayerWiseSampler : public IndexSampler {
 public:
  virtual ~LayerWiseSampler() {}
  explicit LayerWiseSampler(const std::string& name) {
    tree_ = IndexWrapper::GetInstance()->get_tree_index(name);
  }

  void init_layerwise_conf(const std::vector<uint16_t>& layer_sample_counts,
                           uint16_t start_sample_layer,
                           uint16_t seed) override {
    seed_ = seed;
    start_sample_layer_ = start_sample_layer;

    PADDLE_ENFORCE_GT(
        start_sample_layer_,
        0,
        common::errors::InvalidArgument(
            "start sampler layer = [%d], it should greater than 0.",
            start_sample_layer_));
    PADDLE_ENFORCE_LT(start_sample_layer_,
                      tree_->Height(),
                      common::errors::InvalidArgument(
                          "start sampler layer = [%d], it should less than "
                          "max_layer, which is [%d].",
                          start_sample_layer_,
                          tree_->Height()));

    size_t i = 0;
    layer_counts_sum_ = 0;
    layer_counts_.clear();
    int cur_layer = start_sample_layer_;
    while (cur_layer < tree_->Height()) {
      int layer_sample_num = 1;
      if (i < layer_sample_counts.size()) {
        layer_sample_num = layer_sample_counts[i];
      }
      layer_counts_sum_ += layer_sample_num + 1;
      layer_counts_.push_back(layer_sample_num);
      VLOG(3) << "[INFO] level " << cur_layer
              << " sample_layer_counts.push_back: " << layer_sample_num;
      cur_layer += 1;
      i += 1;
    }
    reverse(layer_counts_.begin(), layer_counts_.end());
    VLOG(3) << "sample counts sum: " << layer_counts_sum_;

    auto max_layer = tree_->Height();
    sampler_vec_.clear();
    layer_ids_.clear();

    auto layer_index = max_layer - 1;
    size_t idx = 0;
    while (layer_index >= start_sample_layer_) {
      auto layer_codes = tree_->GetLayerCodes(layer_index);
      layer_ids_.push_back(tree_->GetNodes(layer_codes));
      auto sampler_temp = std::make_shared<phi::math::UniformSampler>(
          layer_ids_[idx].size() - 1, seed_);
      sampler_vec_.push_back(sampler_temp);
      layer_index--;
      idx++;
    }
  }
  std::vector<std::vector<uint64_t>> sample(
      const std::vector<std::vector<uint64_t>>& user_inputs,
      const std::vector<uint64_t>& target_ids,
      bool with_hierarchy) override;

  void sample_from_dataset(
      const uint16_t sample_slot,
      std::vector<paddle::framework::Record>* src_datas,
      std::vector<paddle::framework::Record>* sample_results) override;

 private:
  std::vector<int> layer_counts_;
  int64_t layer_counts_sum_{0};
  std::shared_ptr<TreeIndex> tree_{nullptr};
  int seed_{0};
  int start_sample_layer_{1};
  std::vector<std::shared_ptr<phi::math::Sampler>> sampler_vec_;
  std::vector<std::vector<IndexNode>> layer_ids_;
};

}  // namespace distributed
}  // namespace paddle
