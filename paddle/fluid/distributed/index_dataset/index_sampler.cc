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

#include "paddle/fluid/distributed/index_dataset/index_sampler.h"
#include "paddle/fluid/operators/math/sampler.h"

namespace paddle {
namespace distributed {

using Sampler = paddle::operators::math::Sampler;

std::vector<std::vector<uint64_t>> LayerWiseSampler::sample(
    const std::vector<std::vector<uint64_t>>& user_inputs,
    const std::vector<uint64_t>& target_ids, bool with_hierarchy) {
  auto input_num = target_ids.size();
  auto user_feature_num = user_inputs[0].size();
  std::vector<std::vector<uint64_t>> outputs(
      input_num * layer_counts_sum_,
      std::vector<uint64_t>(user_feature_num + 2));

  auto max_layer = tree_->Height();
  std::vector<Sampler*> sampler_vec(max_layer - start_sample_layer_);
  std::vector<std::vector<IndexNode>> layer_ids(max_layer -
                                                start_sample_layer_);

  auto layer_index = max_layer - 1;
  size_t idx = 0;
  while (layer_index >= start_sample_layer_) {
    auto layer_codes = tree_->GetLayerCodes(layer_index);
    layer_ids[idx] = tree_->GetNodes(layer_codes);
    sampler_vec[idx] = new paddle::operators::math::UniformSampler(
        layer_ids[idx].size() - 1, seed_);
    layer_index--;
    idx++;
  }

  idx = 0;
  for (size_t i = 0; i < input_num; i++) {
    auto travel_codes =
        tree_->GetTravelCodes(target_ids[i], start_sample_layer_);
    auto travel_path = tree_->GetNodes(travel_codes);
    for (size_t j = 0; j < travel_path.size(); j++) {
      // user
      if (j > 0 && with_hierarchy) {
        auto ancestor_codes =
            tree_->GetAncestorCodes(user_inputs[i], max_layer - j - 1);
        auto hierarchical_user = tree_->GetNodes(ancestor_codes);
        for (int idx_offset = 0; idx_offset <= layer_counts_[j]; idx_offset++) {
          for (size_t k = 0; k < user_feature_num; k++) {
            outputs[idx + idx_offset][k] = hierarchical_user[k].id();
          }
        }
      } else {
        for (int idx_offset = 0; idx_offset <= layer_counts_[j]; idx_offset++) {
          for (size_t k = 0; k < user_feature_num; k++) {
            outputs[idx + idx_offset][k] = user_inputs[i][k];
          }
        }
      }

      // sampler ++
      outputs[idx][user_feature_num] = travel_path[j].id();
      outputs[idx][user_feature_num + 1] = 1.0;
      idx += 1;
      for (int idx_offset = 0; idx_offset < layer_counts_[j]; idx_offset++) {
        int sample_res = 0;
        do {
          sample_res = sampler_vec[j]->Sample();
        } while (layer_ids[j][sample_res].id() == travel_path[j].id());
        outputs[idx + idx_offset][user_feature_num] =
            layer_ids[j][sample_res].id();
        outputs[idx + idx_offset][user_feature_num + 1] = 0;
      }
      idx += layer_counts_[j];
    }
  }
  for (size_t i = 0; i < sampler_vec.size(); i++) {
    delete sampler_vec[i];
  }
  return outputs;
}

}  // end namespace distributed
}  // end namespace paddle
