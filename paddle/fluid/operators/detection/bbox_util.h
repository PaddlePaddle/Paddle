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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

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

template <class T>
void ClipTiledBoxes(const platform::DeviceContext& ctx,
                    const framework::Tensor& im_info,
                    const framework::Tensor& input_boxes,
                    framework::Tensor* out) {
  T* out_data = out->mutable_data<T>(ctx.GetPlace());
  const T* im_info_data = im_info.data<T>();
  const T* input_boxes_data = input_boxes.data<T>();
  T zero(0);
  T im_w = round(im_info_data[1] / im_info_data[2]);
  T im_h = round(im_info_data[0] / im_info_data[2]);
  for (int64_t i = 0; i < input_boxes.numel(); ++i) {
    if (i % 4 == 0) {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_w - 1), zero);
    } else if (i % 4 == 1) {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_h - 1), zero);
    } else if (i % 4 == 2) {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_w - 1), zero);
    } else {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_h - 1), zero);
    }
  }
}

}  // namespace operators
}  // namespace paddle
