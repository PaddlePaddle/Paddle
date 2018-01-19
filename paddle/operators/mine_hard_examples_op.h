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
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

enum MiningType { kNone = 0, kMaxNegative, kHardExample };

template <typename T>
bool SortScoreDescend(const std::pair<float, T>& pair1,
                      const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

inline bool IsEligibleMining(const MiningType mining_type, const int match_idx,
                             const float match_dis,
                             const float neg_dis_threshold) {
  if (mining_type == MiningType::kMaxNegative) {
    return match_idx == -1 && match_dis < neg_dis_threshold;
  } else if (mining_type == MiningType::kHardExample) {
    return true;
  } else {
    return false;
  }
}

MiningType GetMiningType(std::string str) {
  if (str == "max_negative") {
    return MiningType::kMaxNegative;
  } else if (str == "hard_example") {
    return MiningType::kHardExample;
  } else {
    return MiningType::kNone;
  }
}

template <typename DeviceContext, typename T>
class MineHardExamplesKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_cls_loss = ctx.Input<framework::Tensor>("ClsLoss");
    auto* in_loc_loss = ctx.Input<framework::Tensor>("LocLoss");
    auto* in_matched_indics = ctx.Input<framework::Tensor>("MatchIndics");
    auto* in_match_dis = ctx.Input<framework::Tensor>("MatchDis");
    float neg_pos_ratio = ctx.Attr<float>("neg_pos_ratio");
    T neg_dis_threshold = static_cast<T>(ctx.Attr<float>("neg_dis_threshold"));
    int sample_size = ctx.Attr<int>("sample_size");
    MiningType mining_type =
        GetMiningType(ctx.Attr<std::string>("mining_type"));

    auto out_neg_indics = ctx.Output<framework::LoDTensor>("NegIndics");
    auto out_match_indics = ctx.Output<framework::Tensor>("UpdatedMatchIndics");

    framework::Copy(*in_matched_indics, ctx.GetPlace(), out_match_indics);

    int batch_size = in_matched_indics->dims()[0];
    int prior_num = in_matched_indics->dims()[1];

    auto match_indices = framework::EigenMatrix<int>::From(*in_matched_indics);

    auto match_indices_et =
        framework::EigenMatrix<int>::From(*out_match_indics);

    auto match_dis = framework::EigenMatrix<float>::From(*in_match_dis);
    auto cls_loss = framework::EigenMatrix<float>::From(*in_cls_loss);
    auto loc_loss = framework::EigenMatrix<float>::From(*in_loc_loss);

    std::vector<std::vector<int>> all_neg_indices;
    int all_neg_num = 0;
    for (int n = 0; n < batch_size; ++n) {
      std::vector<std::pair<float, size_t>> loss_idx;
      int neg_sel = 0;
      for (int m = 0; m < prior_num; ++m) {
        if (IsEligibleMining(mining_type, match_indices(n, m), match_dis(n, m),
                             neg_dis_threshold)) {
          T loss = cls_loss(n, m);
          if (mining_type == MiningType::kHardExample) {
            loss = cls_loss(n, m) + loc_loss(n, m);
          }
          loss_idx.push_back(std::make_pair(loss, m));
          ++neg_sel;
        }
      }
      if (mining_type == MiningType::kMaxNegative) {
        int num_pos = 0;
        for (int m = 0; m < prior_num; ++m) {
          if (match_indices(n, m) != -1) ++num_pos;
        }
        neg_sel = std::min(static_cast<int>(num_pos * neg_pos_ratio), neg_sel);
      } else if (mining_type == MiningType::kHardExample) {
        neg_sel = std::min(sample_size, neg_sel);
      }
      std::sort(loss_idx.begin(), loss_idx.end(), SortScoreDescend<int>);
      std::set<int> sel_indices;
      std::vector<int> neg_indices;
      for (int n = 0; n < neg_sel; ++n) {
        sel_indices.insert(loss_idx[n].second);
      }

      for (int m = 0; m < prior_num; ++m) {
        if (match_indices(n, m) > -1) {
          if (mining_type == MiningType::kHardExample &&
              sel_indices.find(m) == sel_indices.end()) {
            match_indices_et(n, m) = -1;
          }
        } else {
          if (sel_indices.find(m) != sel_indices.end()) {
            neg_indices.push_back(m);
          }
        }
      }
      all_neg_indices.push_back(neg_indices);
      all_neg_num += neg_indices.size();
    }

    framework::LoD out_neg_indics_lod;
    out_neg_indics_lod.resize(1);
    int neg_offset = 0;
    auto neg_data = out_neg_indics->mutable_data<int>(
        framework::make_ddim({all_neg_num, 1}), ctx.GetPlace());
    out_neg_indics_lod[0].push_back(neg_offset);
    for (auto neg_indices : all_neg_indices) {
      for (auto neg_idx : neg_indices) {
        neg_data[neg_offset++] = neg_idx;
      }
      out_neg_indics_lod[0].push_back(neg_offset);
    }
    out_neg_indics->set_lod(out_neg_indics_lod);
    return;
  }
};
}  // namespace operators

}  // namespace paddle
