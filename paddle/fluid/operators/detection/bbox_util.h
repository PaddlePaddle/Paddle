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
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

static const double kBBoxClipDefault = std::log(1000.0 / 16.0);

struct RangeInitFunctor {
  int start;
  int delta;
  int* out;
  HOSTDEVICE void operator()(size_t i) { out[i] = start + i * delta; }
};

template <typename T>
inline HOSTDEVICE T RoIArea(const T* box, bool pixel_offset = true) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<T>(0.);
  } else {
    const T w = box[2] - box[0];
    const T h = box[3] - box[1];
    if (pixel_offset) {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    } else {
      return w * h;
    }
  }
}

/*
 * transform that computes target bounding-box regression deltas
 * given proposal boxes and ground-truth boxes.
 */
template <typename T>
inline void BoxToDelta(const int box_num, const framework::Tensor& ex_boxes,
                       const framework::Tensor& gt_boxes, const float* weights,
                       const bool normalized, framework::Tensor* box_delta) {
  auto ex_boxes_et = framework::EigenTensor<T, 2>::From(ex_boxes);
  auto gt_boxes_et = framework::EigenTensor<T, 2>::From(gt_boxes);
  auto trg = framework::EigenTensor<T, 2>::From(*box_delta);
  T ex_w, ex_h, ex_ctr_x, ex_ctr_y, gt_w, gt_h, gt_ctr_x, gt_ctr_y;
  for (int64_t i = 0; i < box_num; ++i) {
    ex_w = ex_boxes_et(i, 2) - ex_boxes_et(i, 0) + (normalized == false);
    ex_h = ex_boxes_et(i, 3) - ex_boxes_et(i, 1) + (normalized == false);
    ex_ctr_x = ex_boxes_et(i, 0) + 0.5 * ex_w;
    ex_ctr_y = ex_boxes_et(i, 1) + 0.5 * ex_h;

    gt_w = gt_boxes_et(i, 2) - gt_boxes_et(i, 0) + (normalized == false);
    gt_h = gt_boxes_et(i, 3) - gt_boxes_et(i, 1) + (normalized == false);
    gt_ctr_x = gt_boxes_et(i, 0) + 0.5 * gt_w;
    gt_ctr_y = gt_boxes_et(i, 1) + 0.5 * gt_h;

    trg(i, 0) = (gt_ctr_x - ex_ctr_x) / ex_w;
    trg(i, 1) = (gt_ctr_y - ex_ctr_y) / ex_h;
    trg(i, 2) = std::log(gt_w / ex_w);
    trg(i, 3) = std::log(gt_h / ex_h);

    if (weights) {
      trg(i, 0) = trg(i, 0) / weights[0];
      trg(i, 1) = trg(i, 1) / weights[1];
      trg(i, 2) = trg(i, 2) / weights[2];
      trg(i, 3) = trg(i, 3) / weights[3];
    }
  }
}

template <typename T>
void Gather(const T* in, const int in_stride, const int* index, const int num,
            T* out) {
  const int stride_bytes = in_stride * sizeof(T);
  for (int i = 0; i < num; ++i) {
    int id = index[i];
    memcpy(out + i * in_stride, in + id * in_stride, stride_bytes);
  }
}

template <typename T>
void BboxOverlaps(const framework::Tensor& r_boxes,
                  const framework::Tensor& c_boxes,
                  framework::Tensor* overlaps) {
  auto r_boxes_et = framework::EigenTensor<T, 2>::From(r_boxes);
  auto c_boxes_et = framework::EigenTensor<T, 2>::From(c_boxes);
  auto overlaps_et = framework::EigenTensor<T, 2>::From(*overlaps);
  int r_num = r_boxes.dims()[0];
  int c_num = c_boxes.dims()[0];
  auto zero = static_cast<T>(0.0);
  T r_box_area, c_box_area, x_min, y_min, x_max, y_max, inter_w, inter_h,
      inter_area;
  for (int i = 0; i < r_num; ++i) {
    r_box_area = (r_boxes_et(i, 2) - r_boxes_et(i, 0) + 1) *
                 (r_boxes_et(i, 3) - r_boxes_et(i, 1) + 1);
    for (int j = 0; j < c_num; ++j) {
      c_box_area = (c_boxes_et(j, 2) - c_boxes_et(j, 0) + 1) *
                   (c_boxes_et(j, 3) - c_boxes_et(j, 1) + 1);
      x_min = std::max(r_boxes_et(i, 0), c_boxes_et(j, 0));
      y_min = std::max(r_boxes_et(i, 1), c_boxes_et(j, 1));
      x_max = std::min(r_boxes_et(i, 2), c_boxes_et(j, 2));
      y_max = std::min(r_boxes_et(i, 3), c_boxes_et(j, 3));
      inter_w = std::max(x_max - x_min + 1, zero);
      inter_h = std::max(y_max - y_min + 1, zero);
      inter_area = inter_w * inter_h;
      overlaps_et(i, j) =
          (inter_area == 0.) ? 0 : inter_area /
                                       (r_box_area + c_box_area - inter_area);
    }
  }
}

// Calculate max IoU between each box and ground-truth and
// each row represents one box
template <typename T>
void MaxIoU(const framework::Tensor& iou, framework::Tensor* max_iou) {
  const T* iou_data = iou.data<T>();
  int row = iou.dims()[0];
  int col = iou.dims()[1];
  T* max_iou_data = max_iou->data<T>();
  for (int i = 0; i < row; ++i) {
    const T* v = iou_data + i * col;
    T max_v = *std::max_element(v, v + col);
    max_iou_data[i] = max_v;
  }
}

static void AppendProposals(framework::Tensor* dst, int64_t offset,
                            const framework::Tensor& src) {
  auto* out_data = dst->data();
  auto* to_add_data = src.data();
  size_t size_of_t = framework::DataTypeSize(src.dtype());
  offset *= size_of_t;
  std::memcpy(
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(out_data) + offset),
      to_add_data, src.numel() * size_of_t);
}

template <class T>
void ClipTiledBoxes(const platform::DeviceContext& ctx,
                    const framework::Tensor& im_info,
                    const framework::Tensor& input_boxes,
                    framework::Tensor* out, bool is_scale = true,
                    bool pixel_offset = true) {
  T* out_data = out->mutable_data<T>(ctx.GetPlace());
  const T* im_info_data = im_info.data<T>();
  const T* input_boxes_data = input_boxes.data<T>();
  T offset = pixel_offset ? static_cast<T>(1.0) : 0;
  T zero(0);
  T im_w =
      is_scale ? round(im_info_data[1] / im_info_data[2]) : im_info_data[1];
  T im_h =
      is_scale ? round(im_info_data[0] / im_info_data[2]) : im_info_data[0];
  for (int64_t i = 0; i < input_boxes.numel(); ++i) {
    if (i % 4 == 0) {
      out_data[i] =
          std::max(std::min(input_boxes_data[i], im_w - offset), zero);
    } else if (i % 4 == 1) {
      out_data[i] =
          std::max(std::min(input_boxes_data[i], im_h - offset), zero);
    } else if (i % 4 == 2) {
      out_data[i] =
          std::max(std::min(input_boxes_data[i], im_w - offset), zero);
    } else {
      out_data[i] =
          std::max(std::min(input_boxes_data[i], im_h - offset), zero);
    }
  }
}

// Filter the box with small area
template <class T>
void FilterBoxes(const platform::DeviceContext& ctx,
                 const framework::Tensor* boxes, float min_size,
                 const framework::Tensor& im_info, bool is_scale,
                 framework::Tensor* keep, bool pixel_offset = true) {
  const T* im_info_data = im_info.data<T>();
  const T* boxes_data = boxes->data<T>();
  keep->Resize({boxes->dims()[0]});
  min_size = std::max(min_size, 1.0f);
  int* keep_data = keep->mutable_data<int>(ctx.GetPlace());
  T offset = pixel_offset ? static_cast<T>(1.0) : 0;

  int keep_len = 0;
  for (int i = 0; i < boxes->dims()[0]; ++i) {
    T ws = boxes_data[4 * i + 2] - boxes_data[4 * i] + offset;
    T hs = boxes_data[4 * i + 3] - boxes_data[4 * i + 1] + offset;
    if (pixel_offset) {
      T x_ctr = boxes_data[4 * i] + ws / 2;
      T y_ctr = boxes_data[4 * i + 1] + hs / 2;

      if (is_scale) {
        ws = (boxes_data[4 * i + 2] - boxes_data[4 * i]) / im_info_data[2] + 1;
        hs = (boxes_data[4 * i + 3] - boxes_data[4 * i + 1]) / im_info_data[2] +
             1;
      }
      if (ws >= min_size && hs >= min_size && x_ctr <= im_info_data[1] &&
          y_ctr <= im_info_data[0]) {
        keep_data[keep_len++] = i;
      }
    } else {
      if (ws >= min_size && hs >= min_size) {
        keep_data[keep_len++] = i;
      }
    }
  }
  keep->Resize({keep_len});
}

template <class T>
static void BoxCoder(const platform::DeviceContext& ctx,
                     framework::Tensor* all_anchors,
                     framework::Tensor* bbox_deltas,
                     framework::Tensor* variances, framework::Tensor* proposals,
                     const bool pixel_offset = true) {
  T* proposals_data = proposals->mutable_data<T>(ctx.GetPlace());

  int64_t row = all_anchors->dims()[0];
  int64_t len = all_anchors->dims()[1];

  auto* bbox_deltas_data = bbox_deltas->data<T>();
  auto* anchor_data = all_anchors->data<T>();
  const T* variances_data = nullptr;
  if (variances) {
    variances_data = variances->data<T>();
  }

  T offset = pixel_offset ? static_cast<T>(1.0) : 0;
  for (int64_t i = 0; i < row; ++i) {
    T anchor_width = anchor_data[i * len + 2] - anchor_data[i * len] + offset;
    T anchor_height =
        anchor_data[i * len + 3] - anchor_data[i * len + 1] + offset;

    T anchor_center_x = anchor_data[i * len] + 0.5 * anchor_width;
    T anchor_center_y = anchor_data[i * len + 1] + 0.5 * anchor_height;

    T bbox_center_x = 0, bbox_center_y = 0;
    T bbox_width = 0, bbox_height = 0;

    if (variances) {
      bbox_center_x =
          variances_data[i * len] * bbox_deltas_data[i * len] * anchor_width +
          anchor_center_x;
      bbox_center_y = variances_data[i * len + 1] *
                          bbox_deltas_data[i * len + 1] * anchor_height +
                      anchor_center_y;
      bbox_width = std::exp(std::min<T>(variances_data[i * len + 2] *
                                            bbox_deltas_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(variances_data[i * len + 3] *
                                             bbox_deltas_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
    } else {
      bbox_center_x =
          bbox_deltas_data[i * len] * anchor_width + anchor_center_x;
      bbox_center_y =
          bbox_deltas_data[i * len + 1] * anchor_height + anchor_center_y;
      bbox_width = std::exp(std::min<T>(bbox_deltas_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(bbox_deltas_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
    }

    proposals_data[i * len] = bbox_center_x - bbox_width / 2;
    proposals_data[i * len + 1] = bbox_center_y - bbox_height / 2;
    proposals_data[i * len + 2] = bbox_center_x + bbox_width / 2 - offset;
    proposals_data[i * len + 3] = bbox_center_y + bbox_height / 2 - offset;
  }
  // return proposals;
}

}  // namespace operators
}  // namespace paddle
