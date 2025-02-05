// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

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
static inline bool CompareByScore(ScoreWithID<T> a, ScoreWithID<T> b) {
  return a.score >= b.score;
}

template <typename T>
static inline bool CompareByBatchid(ScoreWithID<T> a, ScoreWithID<T> b) {
  return a.batch_id < b.batch_id;
}

template <typename T, typename Context>
void CollectFpnProposalsOpKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& multi_level_rois,
    const std::vector<const DenseTensor*>& multi_level_scores,
    const paddle::optional<std::vector<const DenseTensor*>>&
        multi_level_rois_num,
    int post_nms_topn,
    DenseTensor* fpn_rois_out,
    DenseTensor* rois_num_out) {
  auto multi_layer_rois = multi_level_rois;

  auto multi_layer_scores = multi_level_scores;
  auto multi_rois_num = multi_level_rois_num
                            ? multi_level_rois_num.get()
                            : std::vector<const DenseTensor*>();
  int num_size = multi_rois_num.size();

  auto* fpn_rois = fpn_rois_out;

  PADDLE_ENFORCE_GE(post_nms_topn,
                    0UL,
                    common::errors::InvalidArgument(
                        "The parameter post_nms_topn must be "
                        "a positive integer. But received post_nms_topn = %d",
                        post_nms_topn));

  // assert that the length of Rois and scores are same
  PADDLE_ENFORCE_EQ(
      multi_layer_rois.size(),
      multi_layer_scores.size(),
      common::errors::InvalidArgument(
          "The number of RoIs and Scores should"
          " be the same. But received number of RoIs is %d, number of Scores "
          "is %d",
          multi_layer_rois.size(),
          multi_layer_scores.size()));
  // Check if the lod information of two phi::DenseTensor is same
  const int num_fpn_level = multi_layer_rois.size();
  std::vector<int> integral_of_all_rois(num_fpn_level + 1, 0);
  for (int i = 0; i < num_fpn_level; ++i) {
    int all_rois = 0;
    if (num_size == 0) {
      auto cur_rois_lod = multi_layer_rois[i]->lod().back();
      all_rois = cur_rois_lod[cur_rois_lod.size() - 1];
    } else {
      const int* cur_rois_num = multi_rois_num[i]->data<int>();
      all_rois = std::accumulate(
          cur_rois_num, cur_rois_num + multi_rois_num[i]->numel(), 0);
    }
    integral_of_all_rois[i + 1] = integral_of_all_rois[i] + all_rois;
  }

  const int batch_size = (num_size == 0)
                             ? multi_layer_rois[0]->lod().back().size() - 1
                             : multi_rois_num[0]->numel();
  // concatenate all fpn rois scores into a list
  // create a vector to store all scores
  std::vector<ScoreWithID<T>> scores_of_all_rois(
      integral_of_all_rois[num_fpn_level], ScoreWithID<T>());
  for (int i = 0; i < num_fpn_level; ++i) {
    const T* cur_level_scores = multi_layer_scores[i]->data<T>();
    int cur_level_num = integral_of_all_rois[i + 1] - integral_of_all_rois[i];
    int cur_batch_id = 0;
    int pre_num = 0;
    for (int j = 0; j < cur_level_num; ++j) {
      if (num_size == 0) {
        auto cur_scores_lod = multi_layer_scores[i]->lod().back();
        if (static_cast<size_t>(j) >= cur_scores_lod[cur_batch_id + 1]) {
          cur_batch_id++;
        }
      } else {
        const int* rois_num_data = multi_rois_num[i]->data<int>();
        if (j >= pre_num + rois_num_data[cur_batch_id]) {
          pre_num += rois_num_data[cur_batch_id];
          cur_batch_id++;
        }
      }
      int cur_index = j + integral_of_all_rois[i];
      scores_of_all_rois[cur_index].score = cur_level_scores[j];
      scores_of_all_rois[cur_index].index = j;
      scores_of_all_rois[cur_index].level = i;
      scores_of_all_rois[cur_index].batch_id = cur_batch_id;
    }
  }
  // keep top post_nms_topn rois
  // sort the rois by the score
  if (post_nms_topn > integral_of_all_rois[num_fpn_level]) {
    post_nms_topn = integral_of_all_rois[num_fpn_level];
  }
  std::stable_sort(
      scores_of_all_rois.begin(), scores_of_all_rois.end(), CompareByScore<T>);
  scores_of_all_rois.resize(post_nms_topn);
  // sort by batch id
  std::stable_sort(scores_of_all_rois.begin(),
                   scores_of_all_rois.end(),
                   CompareByBatchid<T>);
  // create a pointer array
  std::vector<const T*> multi_fpn_rois_data(num_fpn_level);
  for (int i = 0; i < num_fpn_level; ++i) {
    multi_fpn_rois_data[i] = multi_layer_rois[i]->data<T>();
  }
  // initialize the outputs
  fpn_rois->Resize({post_nms_topn, kBoxDim});
  dev_ctx.template Alloc<T>(fpn_rois);
  T* fpn_rois_data = fpn_rois->data<T>();
  std::vector<size_t> lod0(1, 0);
  int cur_batch_id = 0;
  std::vector<int64_t> num_per_batch;
  int pre_idx = 0;
  int cur_num = 0;
  for (int i = 0; i < post_nms_topn; ++i) {
    int cur_fpn_level = scores_of_all_rois[i].level;
    int cur_level_index = scores_of_all_rois[i].index;
    memcpy(fpn_rois_data,
           multi_fpn_rois_data[cur_fpn_level] + cur_level_index * kBoxDim,
           kBoxDim * sizeof(T));
    fpn_rois_data += kBoxDim;
    if (scores_of_all_rois[i].batch_id != cur_batch_id) {
      cur_batch_id = scores_of_all_rois[i].batch_id;
      lod0.emplace_back(i);
      cur_num = i - pre_idx;
      pre_idx = i;
      num_per_batch.emplace_back(cur_num);
    }
  }
  num_per_batch.emplace_back(post_nms_topn - pre_idx);
  if (rois_num_out != nullptr) {
    auto* rois_num = rois_num_out;
    rois_num->Resize({batch_size});
    int* rois_num_data = dev_ctx.template Alloc<int>(rois_num);
    for (int i = 0; i < batch_size; i++) {
      rois_num_data[i] = num_per_batch[i];
    }
  }
  lod0.emplace_back(post_nms_topn);
  phi::LegacyLoD lod;
  lod.emplace_back(lod0);
  fpn_rois->set_lod(lod);
}
}  // namespace phi
