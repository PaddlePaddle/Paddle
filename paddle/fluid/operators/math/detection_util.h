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
#pragma once
#include <map>
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {
template <typename T>
struct BBox {
  BBox(T x_min, T y_min, T x_max, T y_max)
      : x_min(x_min),
        y_min(y_min),
        x_max(x_max),
        y_max(y_max),
        is_difficult(false) {}

  BBox() {}

  T get_width() const { return x_max - x_min; }

  T get_height() const { return y_max - y_min; }

  T get_center_x() const { return (x_min + x_max) / 2; }

  T get_center_y() const { return (y_min + y_max) / 2; }

  T get_area() const { return get_width() * get_height(); }

  // coordinate of bounding box
  T x_min;
  T y_min;
  T x_max;
  T y_max;
  // whether difficult object (e.g. object with heavy occlusion is difficult)
  bool is_difficult;
};
// KNCHW ==> NHWC
// template <typename T>
template <typename T>
void GetBBoxFromPriorData(const T* prior_data, const size_t num_bboxes,
                          std::vector<BBox<T>>& bbox_vec);
template <typename T>
void GetBBoxVarFromPriorData(const T* prior_data, const size_t num,
                             std::vector<std::vector<T>>& var_vec);
template <typename T>
BBox<T> DecodeBBoxWithVar(BBox<T>& prior_bbox,
                          const std::vector<T>& prior_bbox_var,
                          const std::vector<T>& loc_pred_data);
template <typename T1, typename T2>
bool SortScorePairDescend(const std::pair<T1, T2>& pair1,
                          const std::pair<T1, T2>& pair2);
template <typename T>
bool SortScorePairDescend(const std::pair<T, BBox<T>>& pair1,
                          const std::pair<T, BBox<T>>& pair2);
template <typename T>
T jaccard_overlap(const BBox<T>& bbox1, const BBox<T>& bbox2);

template <typename T>
void ApplyNmsFast(const std::vector<BBox<T>>& bboxes, const T* conf_score_data,
                  size_t class_idx, size_t top_k, T conf_threshold,
                  T nms_threshold, size_t num_priors, size_t num_classes,
                  std::vector<size_t>* indices);
template <typename T>
int GetDetectionIndices(
    const T* conf_data, const size_t num_priors, const size_t num_classes,
    const size_t background_label_id, const size_t batch_size,
    const T conf_threshold, const size_t nms_top_k, const T nms_threshold,
    const size_t top_k,
    const std::vector<std::vector<BBox<T>>>& all_decoded_bboxes,
    std::vector<std::map<size_t, std::vector<size_t>>>* all_detection_indices);
template <typename T>
BBox<T> ClipBBox(const BBox<T>& bbox);
template <typename T>
void GetDetectionOutput(
    const T* conf_data, const size_t num_kept, const size_t num_priors,
    const size_t num_classes, const size_t batch_size,
    const std::vector<std::map<size_t, std::vector<size_t>>>& all_indices,
    const std::vector<std::vector<BBox<T>>>& all_decoded_bboxes, T* out_data);
template <typename T>
void GetBBoxFromPriorData(const T* prior_data, const size_t num_bboxes,
                          std::vector<BBox<T>>& bbox_vec) {
  size_t out_offset = bbox_vec.size();
  bbox_vec.resize(bbox_vec.size() + num_bboxes);
  for (size_t i = 0; i < num_bboxes; ++i) {
    BBox<T> bbox;
    bbox.x_min = *(prior_data + i * 8);
    bbox.y_min = *(prior_data + i * 8 + 1);
    bbox.x_max = *(prior_data + i * 8 + 2);
    bbox.y_max = *(prior_data + i * 8 + 3);
    bbox_vec[out_offset + i] = bbox;
  }
}
template <typename T>
void GetBBoxVarFromPriorData(const T* prior_data, const size_t num,
                             std::vector<std::vector<T>>& var_vec) {
  size_t out_offset = var_vec.size();
  var_vec.resize(var_vec.size() + num);
  for (size_t i = 0; i < num; ++i) {
    std::vector<T> var;
    var.push_back(*(prior_data + i * 8 + 4));
    var.push_back(*(prior_data + i * 8 + 5));
    var.push_back(*(prior_data + i * 8 + 6));
    var.push_back(*(prior_data + i * 8 + 7));
    var_vec[out_offset + i] = var;
  }
}
template <typename T>
BBox<T> DecodeBBoxWithVar(BBox<T>& prior_bbox,
                          const std::vector<T>& prior_bbox_var,
                          const std::vector<T>& loc_pred_data) {
  T prior_bbox_width = prior_bbox.get_width();
  T prior_bbox_height = prior_bbox.get_height();
  T prior_bbox_center_x = prior_bbox.get_center_x();
  T prior_bbox_center_y = prior_bbox.get_center_y();

  T decoded_bbox_center_x =
      prior_bbox_var[0] * loc_pred_data[0] * prior_bbox_width +
      prior_bbox_center_x;
  T decoded_bbox_center_y =
      prior_bbox_var[1] * loc_pred_data[1] * prior_bbox_height +
      prior_bbox_center_y;
  T decoded_bbox_width =
      std::exp(prior_bbox_var[2] * loc_pred_data[2]) * prior_bbox_width;
  T decoded_bbox_height =
      std::exp(prior_bbox_var[3] * loc_pred_data[3]) * prior_bbox_height;

  BBox<T> decoded_bbox;
  decoded_bbox.x_min = decoded_bbox_center_x - decoded_bbox_width / 2;
  decoded_bbox.y_min = decoded_bbox_center_y - decoded_bbox_height / 2;
  decoded_bbox.x_max = decoded_bbox_center_x + decoded_bbox_width / 2;
  decoded_bbox.y_max = decoded_bbox_center_y + decoded_bbox_height / 2;

  return decoded_bbox;
}
template <typename T1, typename T2>
bool SortScorePairDescend(const std::pair<T1, T2>& pair1,
                          const std::pair<T1, T2>& pair2) {
  return pair1.first > pair2.first;
}
template <typename T>
T jaccard_overlap(const BBox<T>& bbox1, const BBox<T>& bbox2) {
  if (bbox2.x_min > bbox1.x_max || bbox2.x_max < bbox1.x_min ||
      bbox2.y_min > bbox1.y_max || bbox2.y_max < bbox1.y_min) {
    return 0.0;
  } else {
    T inter_x_min = std::max(bbox1.x_min, bbox2.x_min);
    T inter_y_min = std::max(bbox1.y_min, bbox2.y_min);
    T interX_max = std::min(bbox1.x_max, bbox2.x_max);
    T interY_max = std::min(bbox1.y_max, bbox2.y_max);

    T inter_width = interX_max - inter_x_min;
    T inter_height = interY_max - inter_y_min;
    T inter_area = inter_width * inter_height;

    T bbox_area1 = bbox1.get_area();
    T bbox_area2 = bbox2.get_area();

    return inter_area / (bbox_area1 + bbox_area2 - inter_area);
  }
}

template <typename T>
void ApplyNmsFast(const std::vector<BBox<T>>& bboxes, const T* conf_score_data,
                  size_t class_idx, size_t top_k, T conf_threshold,
                  T nms_threshold, size_t num_priors, size_t num_classes,
                  std::vector<size_t>* indices) {
  std::vector<std::pair<T, size_t>> scores;
  for (size_t i = 0; i < num_priors; ++i) {
    size_t conf_offset = i * num_classes + class_idx;
    if (conf_score_data[conf_offset] > conf_threshold)
      scores.push_back(std::make_pair(conf_score_data[conf_offset], i));
  }
  std::stable_sort(scores.begin(), scores.end(),
                   SortScorePairDescend<T, size_t>);
  if (top_k > 0 && top_k < scores.size()) scores.resize(top_k);
  while (scores.size() > 0) {
    const size_t idx = scores.front().second;
    bool keep = true;
    for (size_t i = 0; i < indices->size(); ++i) {
      if (keep) {
        const size_t saved_idx = (*indices)[i];
        T overlap = jaccard_overlap<T>(bboxes[idx], bboxes[saved_idx]);
        keep = overlap <= nms_threshold;
      } else {
        break;
      }
    }
    if (keep) indices->push_back(idx);
    scores.erase(scores.begin());
  }
}
template <typename T>
int GetDetectionIndices(
    const T* conf_data, const size_t num_priors, const size_t num_classes,
    const size_t background_label_id, const size_t batch_size,
    const T conf_threshold, const size_t nms_top_k, const T nms_threshold,
    const size_t top_k,
    const std::vector<std::vector<BBox<T>>>& all_decoded_bboxes,
    std::vector<std::map<size_t, std::vector<size_t>>>* all_detection_indices) {
  int total_keep_num = 0;
  for (size_t n = 0; n < batch_size; ++n) {
    const std::vector<BBox<T>>& decoded_bboxes = all_decoded_bboxes[n];
    size_t num_detected = 0;
    std::map<size_t, std::vector<size_t>> indices;
    size_t conf_offset = n * num_priors * num_classes;
    for (size_t c = 0; c < num_classes; ++c) {
      if (c == background_label_id) continue;
      ApplyNmsFast<T>(decoded_bboxes, conf_data + conf_offset, c, nms_top_k,
                      conf_threshold, nms_threshold, num_priors, num_classes,
                      &(indices[c]));
      num_detected += indices[c].size();
    }
    if (top_k > 0 && num_detected > top_k) {
      // std::vector<pair<T,T>> score_index_pairs;
      std::vector<std::pair<T, std::pair<size_t, size_t>>> score_index_pairs;
      for (size_t c = 0; c < num_classes; ++c) {
        const std::vector<size_t>& label_indices = indices[c];
        for (size_t i = 0; i < label_indices.size(); ++i) {
          size_t idx = label_indices[i];
          score_index_pairs.push_back(
              std::make_pair((conf_data + conf_offset)[idx * num_classes + c],
                             std::make_pair(c, idx)));
        }
      }
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<T, std::pair<size_t, size_t>>);
      score_index_pairs.resize(top_k);
      std::map<size_t, std::vector<size_t>> new_indices;
      for (size_t i = 0; i < score_index_pairs.size(); ++i) {
        size_t label = score_index_pairs[i].second.first;
        size_t idx = score_index_pairs[i].second.second;
        new_indices[label].push_back(idx);
      }
      all_detection_indices->push_back(new_indices);
      total_keep_num += top_k;
    } else {
      all_detection_indices->push_back(indices);
      total_keep_num += num_detected;
    }
  }
  return total_keep_num;
}
template <typename T>
BBox<T> ClipBBox(const BBox<T>& bbox) {
  T one = static_cast<T>(1.0);
  T zero = static_cast<T>(0.0);
  BBox<T> clipped_bbox;
  clipped_bbox.x_min = std::max(std::min(bbox.x_min, one), zero);
  clipped_bbox.y_min = std::max(std::min(bbox.y_min, one), zero);
  clipped_bbox.x_max = std::max(std::min(bbox.x_max, one), zero);
  clipped_bbox.y_max = std::max(std::min(bbox.y_max, one), zero);
  return clipped_bbox;
}
template <typename T>
void GetDetectionOutput(
    const T* conf_data, const size_t num_kept, const size_t num_priors,
    const size_t num_classes, const size_t batch_size,
    const std::vector<std::map<size_t, std::vector<size_t>>>& all_indices,
    const std::vector<std::vector<BBox<T>>>& all_decoded_bboxes, T* out_data) {
  size_t count = 0;
  for (size_t n = 0; n < batch_size; ++n) {
    for (std::map<size_t, std::vector<size_t>>::const_iterator it =
             all_indices[n].begin();
         it != all_indices[n].end(); ++it) {
      size_t label = it->first;
      const std::vector<size_t>& indices = it->second;
      const std::vector<BBox<T>>& decoded_bboxes = all_decoded_bboxes[n];
      for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        size_t conf_offset = n * num_priors * num_classes + idx * num_classes;
        out_data[count * 7] = n;
        out_data[count * 7 + 1] = label;
        out_data[count * 7 + 2] = (conf_data + conf_offset)[label];
        BBox<T> clipped_bbox = ClipBBox<T>(decoded_bboxes[idx]);
        out_data[count * 7 + 3] = clipped_bbox.x_min;
        out_data[count * 7 + 4] = clipped_bbox.y_min;
        out_data[count * 7 + 5] = clipped_bbox.x_max;
        out_data[count * 7 + 6] = clipped_bbox.y_max;
        ++count;
      }
    }
  }
}
}  // namespace math
}  // namespace operators
}  // namespace paddle
