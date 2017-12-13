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
#include "paddle/framework/selected_rows.h"
#include "paddle/platform/device_context.h"

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

template <typename T>
void GetBBoxFromDetectData(const T* detect_data, const size_t num_bboxes,
                           std::vector<T>& labels, std::vector<T>& scores,
                           std::vector<BBox<T>>& bboxes) {
  size_t out_offset = bboxes.size();
  labels.resize(out_offset + num_bboxes);
  scores.resize(out_offset + num_bboxes);
  bboxes.resize(out_offset + num_bboxes);
  for (size_t i = 0; i < num_bboxes; ++i) {
    labels[out_offset + i] = *(detect_data + i * 7 + 1);
    scores[out_offset + i] = *(detect_data + i * 7 + 2);
    BBox<T> bbox;
    bbox.x_min = *(detect_data + i * 7 + 3);
    bbox.y_min = *(detect_data + i * 7 + 4);
    bbox.x_max = *(detect_data + i * 7 + 5);
    bbox.y_max = *(detect_data + i * 7 + 6);
    bboxes[out_offset + i] = bbox;
  };
}

template <typename T>
void GetBBoxFromLabelData(const T* label_data, const size_t num_bboxes,
                          std::vector<BBox<T>>& bboxes) {
  size_t out_offset = bboxes.size();
  bboxes.resize(bboxes.size() + num_bboxes);
  for (size_t i = 0; i < num_bboxes; ++i) {
    BBox<T> bbox;
    bbox.x_min = *(label_data + i * 6 + 1);
    bbox.y_min = *(label_data + i * 6 + 2);
    bbox.x_max = *(label_data + i * 6 + 3);
    bbox.y_max = *(label_data + i * 6 + 4);
    T is_difficult = *(label_data + i * 6 + 5);
    if (std::abs(is_difficult - 0.0) < 1e-6)
      bbox.is_difficult = false;
    else
      bbox.is_difficult = true;
    bboxes[out_offset + i] = bbox;
  }
}

template <typename T>
inline float JaccardOverlap(const BBox<T>& bbox1, const BBox<T>& bbox2) {
  if (bbox2.x_min > bbox1.x_max || bbox2.x_max < bbox1.x_min ||
      bbox2.y_min > bbox1.y_max || bbox2.y_max < bbox1.y_min) {
    return 0.0;
  } else {
    float inter_x_min = std::max(bbox1.x_min, bbox2.x_min);
    float inter_y_min = std::max(bbox1.y_min, bbox2.y_min);
    float inter_x_max = std::min(bbox1.x_max, bbox2.x_max);
    float inter_y_max = std::min(bbox1.y_max, bbox2.y_max);

    float inter_width = inter_x_max - inter_x_min;
    float inter_height = inter_y_max - inter_y_min;
    float inter_area = inter_width * inter_height;

    float bbox_area1 = bbox1.get_area();
    float bbox_area2 = bbox2.get_area();

    return inter_area / (bbox_area1 + bbox_area2 - inter_area);
  }
}

template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

// template <>
// bool SortScorePairDescend(const std::pair<float, NormalizedBBox>& pair1,
//                           const std::pair<float, NormalizedBBox>& pair2) {
//   return pair1.first > pair2.first;
// }

}  // namespace math
}  // namespace operators
}  // namespace paddle
