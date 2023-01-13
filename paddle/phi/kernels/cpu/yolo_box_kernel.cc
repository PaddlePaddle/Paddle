// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/yolo_box_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/yolo_box_util.h"

namespace phi {

template <typename T, typename Context>
void YoloBoxKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& img_size,
                   const std::vector<int>& anchors,
                   int class_num,
                   float conf_thresh,
                   int downsample_ratio,
                   bool clip_bbox,
                   float scale_x_y,
                   bool iou_aware,
                   float iou_aware_factor,
                   DenseTensor* boxes,
                   DenseTensor* scores) {
  auto* input = &x;
  auto* imgsize = &img_size;
  float scale = scale_x_y;
  float bias = -0.5 * (scale - 1.);

  const int n = input->dims()[0];
  const int h = input->dims()[2];
  const int w = input->dims()[3];
  const int box_num = boxes->dims()[1];
  const int an_num = anchors.size() / 2;
  int input_size_h = downsample_ratio * h;
  int input_size_w = downsample_ratio * w;

  const int stride = h * w;
  const int an_stride = (class_num + 5) * stride;

  DenseTensor anchors_;
  anchors_.Resize({an_num * 2});
  auto anchors_data = dev_ctx.template Alloc<int>(&anchors_);
  std::copy(anchors.begin(), anchors.end(), anchors_data);

  const T* input_data = input->data<T>();
  const int* imgsize_data = imgsize->data<int>();
  boxes->Resize({n, box_num, 4});
  T* boxes_data = dev_ctx.template Alloc<T>(boxes);
  memset(boxes_data, 0, boxes->numel() * sizeof(T));

  scores->Resize({n, box_num, class_num});
  T* scores_data = dev_ctx.template Alloc<T>(scores);

  memset(scores_data, 0, scores->numel() * sizeof(T));

  T box[4];
  for (int i = 0; i < n; i++) {
    int img_height = imgsize_data[2 * i];
    int img_width = imgsize_data[2 * i + 1];

    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < h; k++) {
        for (int l = 0; l < w; l++) {
          int obj_idx = funcs::GetEntryIndex(
              i, j, k * w + l, an_num, an_stride, stride, 4, iou_aware);
          T conf = funcs::sigmoid<T>(input_data[obj_idx]);
          if (iou_aware) {
            int iou_idx =
                funcs::GetIoUIndex(i, j, k * w + l, an_num, an_stride, stride);
            T iou = funcs::sigmoid<T>(input_data[iou_idx]);
            conf = pow(conf, static_cast<T>(1. - iou_aware_factor)) *
                   pow(iou, static_cast<T>(iou_aware_factor));
          }
          if (conf < conf_thresh) {
            continue;
          }

          int box_idx = funcs::GetEntryIndex(
              i, j, k * w + l, an_num, an_stride, stride, 0, iou_aware);
          funcs::GetYoloBox<T>(box,
                               input_data,
                               anchors_data,
                               l,
                               k,
                               j,
                               h,
                               w,
                               input_size_h,
                               input_size_w,
                               box_idx,
                               stride,
                               img_height,
                               img_width,
                               scale,
                               bias);
          box_idx = (i * box_num + j * stride + k * w + l) * 4;
          funcs::CalcDetectionBox<T>(
              boxes_data, box, box_idx, img_height, img_width, clip_bbox);

          int label_idx = funcs::GetEntryIndex(
              i, j, k * w + l, an_num, an_stride, stride, 5, iou_aware);
          int score_idx = (i * box_num + j * stride + k * w + l) * class_num;
          funcs::CalcLabelScore<T>(scores_data,
                                   input_data,
                                   label_idx,
                                   score_idx,
                                   class_num,
                                   conf,
                                   stride);
        }
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    yolo_box, CPU, ALL_LAYOUT, phi::YoloBoxKernel, float, double) {}
