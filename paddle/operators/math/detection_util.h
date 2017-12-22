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
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/strided_memcpy.h"
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
void GetBBoxFromPriorData(const T* prior_data, const size_t num_bboxes,
                          std::vector<BBox<T>>& bboxes) {
  size_t out_offset = bboxes.size();
  bboxes.resize(bboxes.size() + num_bboxes);
  for (size_t i = 0; i < num_bboxes; ++i) {
    BBox<T> bbox;
    bbox.x_min = *(prior_data + i * 8);
    bbox.y_min = *(prior_data + i * 8 + 1);
    bbox.x_max = *(prior_data + i * 8 + 2);
    bbox.y_max = *(prior_data + i * 8 + 3);
    bboxes[out_offset + i] = bbox;
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
void EncodeBBoxWithVar(const BBox<T>& prior_bbox,
                       const std::vector<T>& prior_bbox_var,
                       const BBox<T>& gt_bbox, std::vector<T>& out_vec) {
  T prior_bbox_width = prior_bbox.get_width();
  T prior_bbox_height = prior_bbox.get_height();
  T prior_bbox_center_x = prior_bbox.get_center_x();
  T prior_bbox_center_y = prior_bbox.get_center_y();

  T gt_bbox_width = gt_bbox.get_width();
  T gt_bbox_height = gt_bbox.get_height();
  T gt_bbox_center_x = gt_bbox.get_center_x();
  T gt_bbox_center_y = gt_bbox.get_center_y();

  out_vec.clear();
  out_vec.push_back((gt_bbox_center_x - prior_bbox_center_x) /
                    prior_bbox_width / prior_bbox_var[0]);
  out_vec.push_back((gt_bbox_center_y - prior_bbox_center_y) /
                    prior_bbox_height / prior_bbox_var[1]);
  out_vec.push_back(std::log(std::fabs(gt_bbox_width / prior_bbox_width)) /
                    prior_bbox_var[2]);
  out_vec.push_back(std::log(std::fabs(gt_bbox_height / prior_bbox_height)) /
                    prior_bbox_var[3]);
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
void MatchBBox(const std::vector<BBox<T>>& prior_bboxes,
               const std::vector<BBox<T>>& gt_bboxes, float overlap_threshold,
               std::vector<int>& match_indices,
               std::vector<float>& match_overlaps) {
  std::map<size_t, std::map<size_t, float>> overlaps;
  size_t num_priors = prior_bboxes.size();
  size_t num_gts = gt_bboxes.size();

  match_indices.clear();
  match_indices.resize(num_priors, -1);
  match_overlaps.clear();
  match_overlaps.resize(num_priors, 0.0);

  // Store the positive overlap between predictions and ground truth
  for (size_t i = 0; i < num_priors; ++i) {
    for (size_t j = 0; j < num_gts; ++j) {
      float overlap = JaccardOverlap<T>(prior_bboxes[i], gt_bboxes[j]);
      if (overlap > 1e-6) {
        match_overlaps[i] = std::max(match_overlaps[i], overlap);
        overlaps[i][j] = overlap;
      }
    }
  }
  // Bipartite matching
  std::vector<int> gt_pool;
  for (size_t i = 0; i < num_gts; ++i) {
    gt_pool.push_back(i);
  }
  while (gt_pool.size() > 0) {
    // Find the most overlapped gt and corresponding predictions
    int max_prior_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1.0;
    for (auto it = overlaps.begin(); it != overlaps.end(); ++it) {
      size_t i = it->first;
      if (match_indices[i] != -1) {
        // The prediction already has matched ground truth or is ignored
        continue;
      }
      for (size_t p = 0; p < gt_pool.size(); ++p) {
        int j = gt_pool[p];
        if (it->second.find(j) == it->second.end()) {
          // No overlap between the i-th prediction and j-th ground truth
          continue;
        }
        // Find the maximum overlapped pair
        if (it->second[j] > max_overlap) {
          max_prior_idx = (int)i;
          max_gt_idx = (int)j;
          max_overlap = it->second[j];
        }
      }
    }
    if (max_prior_idx == -1) {
      break;
    } else {
      match_indices[max_prior_idx] = max_gt_idx;
      match_overlaps[max_prior_idx] = max_overlap;
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  // Get most overlaped for the rest prediction bboxes
  for (auto it = overlaps.begin(); it != overlaps.end(); ++it) {
    size_t i = it->first;
    if (match_indices[i] != -1) {
      // The prediction already has matched ground truth or is ignored
      continue;
    }
    int max_gt_idx = -1;
    float max_overlap = -1;
    for (size_t j = 0; j < num_gts; ++j) {
      if (it->second.find(j) == it->second.end()) {
        // No overlap between the i-th prediction and j-th ground truth
        continue;
      }
      // Find the maximum overlapped pair
      float overlap = it->second[j];
      if (overlap > max_overlap && overlap >= overlap_threshold) {
        max_gt_idx = j;
        max_overlap = overlap;
      }
    }
    if (max_gt_idx != -1) {
      match_indices[i] = max_gt_idx;
      match_overlaps[i] = max_overlap;
    }
  }
}

template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template <typename DeviceContext, typename T>
int TransposeFromNCHWToNHWC(const DeviceContext& ctx,
                            const framework::Tensor& src,
                            framework::Tensor& dst, int dst_total_size,
                            int dst_offset) {
  int batch_size = src.dims()[0];
  std::vector<int64_t> shape_vec(
      {src.dims()[0], src.dims()[2], src.dims()[3], src.dims()[1]});
  auto shape = framework::make_ddim(shape_vec);
  framework::Tensor src_transpose;
  src_transpose.mutable_data<T>(shape, ctx.GetPlace());
  std::vector<int> shape_axis({0, 2, 3, 1});
  math::Transpose<DeviceContext, T, 4> trans4;
  trans4(ctx, src, &src_transpose, shape_axis);

  auto src_stride = framework::stride(src.dims());

  for (int i = 0; i < batch_size; ++i) {
    int out_offset = i * (dst_total_size / batch_size) + dst_offset;
    framework::Tensor src_i = src_transpose.Slice(i, i + 1);

    src_i.Resize(framework::make_ddim({1, src_i.numel()}));

    StridedMemcpy<T>(ctx, src_i.data<T>(), framework::stride(src_i.dims()),
                     src_i.dims(), framework::stride(dst.dims()),
                     dst.data<T>() + out_offset);
  }

  return src_stride[0];
}

template <typename DeviceContext, typename T>
int TransposeFromNHWCToNCHW(const DeviceContext& ctx,
                            const framework::Tensor& src, int src_total_size,
                            int src_offset, framework::Tensor& dst) {
  int batch_size = dst.dims()[0];

  framework::Tensor dst_transpose;
  std::vector<int64_t> shape_vec(
      {dst.dims()[0], dst.dims()[3], dst.dims()[1], dst.dims()[2]});
  dst_transpose.mutable_data<T>(framework::make_ddim(shape_vec),
                                ctx.GetPlace());
  auto dst_stride = framework::stride(dst.dims());

  for (int i = 0; i < batch_size; ++i) {
    int in_offset = i * (src_total_size / batch_size) + src_offset;
    framework::Tensor dst_i = dst_transpose.Slice(i, i + 1);

    dst_i.Resize(framework::make_ddim({1, dst_stride[0]}));
    StridedMemcpy<T>(ctx, src.data<T>() + in_offset,
                     framework::stride(src.dims()), src.dims(),
                     framework::stride(dst_i.dims()), dst_i.data<T>());
  }

  std::vector<int> shape_axis({0, 3, 1, 2});
  math::Transpose<DeviceContext, T, 4> trans4;
  trans4(ctx, dst_transpose, &dst, shape_axis);
  return dst_stride[0];
}

}  // namespace math

}  // namespace operators
}  // namespace paddle
