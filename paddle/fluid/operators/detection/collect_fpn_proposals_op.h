/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

const int kBoxDim = 4;

template <typename T>
struct ScoreWithID {
  T score;
  int batch_id;
  int index;
  int level;
  ScoreWithID() {
    batch_id = -1;
    index = -1;
    level = -1;
  }
  ScoreWithID(T score_, int batch_id_, int index_, int level_) {
    score = score_;
    batch_id = batch_id_;
    index = index_;
    level = level_;
  }
};
template <typename T>
bool compare_by_score(ScoreWithID<T> a, ScoreWithID<T> b) {
  return a.score >= b.score;
}

template <typename T>
bool compare_by_batchid(ScoreWithID<T> a, ScoreWithID<T> b) {
  return a.batch_id < b.batch_id;
}

template <typename T>
class CollectFpnProposalsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto multi_layer_rois =
        context.MultiInput<paddle::framework::LoDTensor>("MultiLayerRois");

    auto multi_layer_scores =
        context.MultiInput<paddle::framework::LoDTensor>("MultiLayerScores");

    auto* fpn_rois = context.Output<paddle::framework::LoDTensor>("FpnRois");

    int post_nms_topN = context.Attr<int>("post_nms_topN");

    PADDLE_ENFORCE_GE(post_nms_topN, 0UL,
                      "The parameter post_nms_topN must be a positive integer");

    // assert that the length of Rois and scores are same
    PADDLE_ENFORCE(multi_layer_rois.size() == multi_layer_scores.size(),
                   "DistributeFpnProposalsOp need 1 level of LoD");
    // Check if the lod information of two LoDTensor is same
    const int num_fpn_level = multi_layer_rois.size();
    std::vector<int> integral_of_all_rois(num_fpn_level + 1, 0);
    for (int i = 0; i < num_fpn_level; ++i) {
      auto cur_rois_lod = multi_layer_rois[i]->lod().back();
      auto cur_scores_lod = multi_layer_scores[i]->lod().back();
      PADDLE_ENFORCE(cur_rois_lod.size() == cur_scores_lod.size(),
                     "The length of FPN Rois LoD does not match the Length of "
                     "FPN Scores LoD");
      for (int j = 0; j < cur_rois_lod.size(); ++j) {
        PADDLE_ENFORCE(cur_rois_lod[j] == cur_scores_lod[j],
                       "The LoD of FPN Rois is not equal to LoD of FPN Scores");
      }
      integral_of_all_rois[i + 1] =
          integral_of_all_rois[i] + cur_rois_lod[cur_rois_lod.size() - 1];
    }

    // concatenate all fpn rois scores into a list
    // create a vector to store all scores
    std::vector<ScoreWithID<T>> scores_of_all_rois(
        integral_of_all_rois[num_fpn_level], ScoreWithID<T>());
    for (int i = 0; i < num_fpn_level; ++i) {
      const T* cur_level_scores = multi_layer_scores[i]->data<T>();
      int cur_level_num = integral_of_all_rois[i + 1] - integral_of_all_rois[i];
      auto cur_scores_lod = multi_layer_scores[i]->lod().back();
      int cur_batch_id = 0;
      for (int j = 0; j < cur_level_num; ++j) {
        if (j >= cur_scores_lod[cur_batch_id + 1]) {
          cur_batch_id++;
        }
        int cur_index = j + integral_of_all_rois[i];
        scores_of_all_rois[cur_index].score = cur_level_scores[j];
        scores_of_all_rois[cur_index].index = j;
        scores_of_all_rois[cur_index].level = i;
        scores_of_all_rois[cur_index].batch_id = cur_batch_id;
      }
    }
    // keep top post_nms_topN rois
    // sort the rois by the score
    if (post_nms_topN > integral_of_all_rois[num_fpn_level]) {
      post_nms_topN = integral_of_all_rois[num_fpn_level];
    }
    std::stable_sort(scores_of_all_rois.begin(), scores_of_all_rois.end(),
                     compare_by_score<T>);
    scores_of_all_rois.resize(post_nms_topN);
    // sort by batch id
    std::stable_sort(scores_of_all_rois.begin(), scores_of_all_rois.end(),
                     compare_by_batchid<T>);
    // create a pointer array
    std::vector<const T*> multi_fpn_rois_data(num_fpn_level);
    for (int i = 0; i < num_fpn_level; ++i) {
      multi_fpn_rois_data[i] = multi_layer_rois[i]->data<T>();
    }
    // initialize the outputs
    fpn_rois->mutable_data<T>({post_nms_topN, kBoxDim}, context.GetPlace());
    T* fpn_rois_data = fpn_rois->data<T>();
    std::vector<size_t> lod0(1, 0);
    int cur_batch_id = 0;
    for (int i = 0; i < post_nms_topN; ++i) {
      int cur_fpn_level = scores_of_all_rois[i].level;
      int cur_level_index = scores_of_all_rois[i].index;
      memcpy(fpn_rois_data,
             multi_fpn_rois_data[cur_fpn_level] + cur_level_index * kBoxDim,
             kBoxDim * sizeof(T));
      fpn_rois_data += kBoxDim;
      if (scores_of_all_rois[i].batch_id != cur_batch_id) {
        cur_batch_id = scores_of_all_rois[i].batch_id;
        lod0.emplace_back(i);
      }
    }
    lod0.emplace_back(post_nms_topN);
    framework::LoD lod;
    lod.emplace_back(lod0);
    fpn_rois->set_lod(lod);
  }
};
}  // namespace operators
}  // namespace paddle
