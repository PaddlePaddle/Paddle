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
                       const framework::Tensor& gt_boxes, const T* weights,
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

}  // namespace operators
}  // namespace paddle
