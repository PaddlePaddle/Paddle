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

#include "paddle/phi/kernels/yolo_box_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
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
  using XPUType = typename XPUTypeTrait<T>::Type;
  int r = 0;
  auto* input = &x;
  // auto* imgsize = &img_size;
  float scale = scale_x_y;
  float bias = -0.5f * (scale - 1.f);

  const int n = static_cast<int>(input->dims()[0]);
  const int h = static_cast<int>(input->dims()[2]);
  const int w = static_cast<int>(input->dims()[3]);
  const int box_num = static_cast<int>(boxes->dims()[1]);
  const int an_num = static_cast<int>(anchors.size() / 2);

  boxes->Resize({n, box_num, 4});
  dev_ctx.template Alloc<T>(boxes);

  scores->Resize({n, box_num, class_num});
  dev_ctx.template Alloc<T>(scores);

  auto x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto img_size_data = reinterpret_cast<const int*>(img_size.data<int>());
  auto boxes_data = reinterpret_cast<XPUType*>(boxes->data<T>());
  auto scores_data = reinterpret_cast<XPUType*>(scores->data<T>());

  std::vector<int64_t> anchors_int64;
  anchors_int64.resize(anchors.size());
  for (size_t i = 0; i < anchors.size(); ++i) {
    anchors_int64[i] = anchors[i];
  }

  r = xpu::yolo_box<float>(dev_ctx.x_context(),
                           x_data,
                           img_size_data,
                           boxes_data,
                           scores_data,
                           n,
                           h,
                           w,
                           anchors_int64,
                           an_num,
                           class_num,
                           conf_thresh,
                           downsample_ratio,
                           scale,
                           bias,
                           false);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "yolo_box");
}

}  // namespace phi

PD_REGISTER_KERNEL(yolo_box, XPU, ALL_LAYOUT, phi::YoloBoxKernel, float) {}
