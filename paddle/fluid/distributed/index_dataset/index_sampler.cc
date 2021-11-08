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
#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace distributed {

void LayerWiseSampler::sample(
    const uint16_t sample_slot,
    std::vector<paddle::framework::Record>* src_datas,
    std::vector<paddle::framework::Record>* sample_results) {
  sample_results->clear();
  for (auto& data : *src_datas) {
    VLOG(1) << "src data size = " << src_datas->size();
    VLOG(1) << "float data size = " << data.float_feasigns_.size();
    // data.Print();
    uint64_t start_idx = sample_results->size();
    VLOG(1) << "before sample, sample_results.size = " << start_idx;
    uint64_t sample_feasign_idx = -1;
    bool sample_sign = false;
    for (unsigned int i = 0; i < data.uint64_feasigns_.size(); i++) {
      VLOG(1) << "slot" << i << " = " << data.uint64_feasigns_[i].slot();
      if (data.uint64_feasigns_[i].slot() == sample_slot) {
        sample_sign = true;
        sample_feasign_idx = i;
      }
      if (sample_sign) break;
    }

    VLOG(1) << "sample_feasign_idx: " << sample_feasign_idx;
    if (sample_sign) {
      auto target_id =
          data.uint64_feasigns_[sample_feasign_idx].sign().uint64_feasign_;
      auto travel_codes = tree_->GetTravelCodes(target_id, start_sample_layer_);
      auto travel_path = tree_->GetNodes(travel_codes);
      for (unsigned int j = 0; j < travel_path.size(); j++) {
        paddle::framework::Record instance(data);
        instance.uint64_feasigns_[sample_feasign_idx].sign().uint64_feasign_ =
            travel_path[j].id();
        sample_results->push_back(instance);
        for (int idx_offset = 0; idx_offset <= layer_counts_[j]; idx_offset++) {
          int sample_res = 0;
          do {
            sample_res = sampler_vec_[j]->Sample();
          } while (layer_ids_[j][sample_res].id() == travel_path[j].id());
          paddle::framework::Record instance(data);
          instance.uint64_feasigns_[sample_feasign_idx].sign().uint64_feasign_ =
              layer_ids_[j][sample_res].id();
          VLOG(1) << "layer id :" << layer_ids_[j][sample_res].id();
          // sample_feasign_idx + 1 == label's id
          instance.uint64_feasigns_[sample_feasign_idx + 1]
              .sign()
              .uint64_feasign_ = 0;
          sample_results->push_back(instance);
        }
        VLOG(1) << "layer end!!!!!!!!!!!!!!!!!!";
      }
    }
  }
  VLOG(1) << "after sample, sample_results.size = " << sample_results->size();
  return;
}

std::vector<uint64_t> float2int(std::vector<double> tmp) {
  std::vector<uint64_t> tmp_int;
  for (auto i : tmp) tmp_int.push_back(uint64_t(i));
  return tmp_int;
}

}  // end namespace distributed
}  // end namespace paddle
