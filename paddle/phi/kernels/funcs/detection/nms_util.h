/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/funcs/detection/poly_util.h"

namespace phi {
namespace funcs {

template <class T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template <class T>
static inline void GetMaxScoreIndex(
    const std::vector<T>& scores,
    const T threshold,
    int top_k,
    std::vector<std::pair<T, int>>* sorted_indices) {
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      sorted_indices->push_back(std::make_pair(scores[i], i));
    }
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices->begin(),
                   sorted_indices->end(),
                   SortScorePairDescend<int>);
  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(top_k);
  }
}

template <class T>
static inline T BBoxArea(const T* box, const bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<T>(0.);
  } else {
    const T w = box[2] - box[0];
    const T h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

template <class T>
static inline T JaccardOverlap(const T* box1,
                               const T* box2,
                               const bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = std::max(box1[0], box2[0]);
    const T inter_ymin = std::max(box1[1], box2[1]);
    const T inter_xmax = std::min(box1[2], box2[2]);
    const T inter_ymax = std::min(box1[3], box2[3]);
    T norm = normalized ? static_cast<T>(0.) : static_cast<T>(1.);
    T inter_w = inter_xmax - inter_xmin + norm;
    T inter_h = inter_ymax - inter_ymin + norm;
    const T inter_area = inter_w * inter_h;
    const T bbox1_area = BBoxArea<T>(box1, normalized);
    const T bbox2_area = BBoxArea<T>(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <class T>
T PolyIoU(const T* box1,
          const T* box2,
          const size_t box_size,
          const bool normalized) {
  T bbox1_area = phi::funcs::PolyArea<T>(box1, box_size, normalized);
  T bbox2_area = phi::funcs::PolyArea<T>(box2, box_size, normalized);
  T inter_area =
      phi::funcs::PolyOverlapArea<T>(box1, box2, box_size, normalized);
  if (bbox1_area == 0 || bbox2_area == 0 || inter_area == 0) {
    // If coordinate values are invalid
    // if area size <= 0,  return 0.
    return T(0.);
  } else {
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <class T>
static inline std::vector<std::pair<T, int>> GetSortedScoreIndex(
    const std::vector<T>& scores) {
  std::vector<std::pair<T, int>> sorted_indices;
  sorted_indices.reserve(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    sorted_indices.emplace_back(scores[i], i);
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices.begin(),
                   sorted_indices.end(),
                   [](const std::pair<T, int>& a, const std::pair<T, int>& b) {
                     return a.first < b.first;
                   });
  return sorted_indices;
}

template <typename T>
static inline DenseTensor VectorToTensor(const DeviceContext& ctx,
                                         const std::vector<T>& selected_indices,
                                         int selected_num) {
  DenseTensor keep_nms;
  keep_nms.Resize({selected_num});
  auto* keep_data = ctx.template Alloc<T>(&keep_nms);
  for (int i = 0; i < selected_num; ++i) {
    keep_data[i] = selected_indices[i];
  }
  return keep_nms;
}

template <class T>
DenseTensor NMS(const DeviceContext& ctx,
                DenseTensor* bbox,
                DenseTensor* scores,
                T nms_threshold,
                float eta,
                bool pixel_offset = true) {
  int64_t num_boxes = bbox->dims()[0];
  // 4: [xmin ymin xmax ymax]
  int64_t box_size = bbox->dims()[1];

  std::vector<T> scores_data(num_boxes);
  std::copy_n(scores->data<T>(), num_boxes, scores_data.begin());
  std::vector<std::pair<T, int>> sorted_indices =
      GetSortedScoreIndex<T>(scores_data);

  std::vector<int> selected_indices;
  int selected_num = 0;
  T adaptive_threshold = nms_threshold;
  const T* bbox_data = bbox->data<T>();
  bool normalized = pixel_offset ? false : true;
  while (sorted_indices.size() != 0) {
    int idx = sorted_indices.back().second;
    bool flag = true;
    for (int kept_idx : selected_indices) {
      if (flag) {
        T overlap = JaccardOverlap<T>(bbox_data + idx * box_size,
                                      bbox_data + kept_idx * box_size,
                                      normalized);
        flag = (overlap <= adaptive_threshold);
      } else {
        break;
      }
    }
    if (flag) {
      selected_indices.push_back(idx);
      ++selected_num;
    }
    sorted_indices.erase(sorted_indices.end() - 1);
    if (flag && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
  return VectorToTensor(ctx, selected_indices, selected_num);
}

}  // namespace funcs
}  // namespace phi
