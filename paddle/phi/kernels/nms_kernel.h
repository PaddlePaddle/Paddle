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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/hostdevice.h"

namespace phi {

HOSTDEVICE static inline int64_t CeilDivide(int64_t n, int64_t m) {
  return (n + m - 1) / m;
}

template <typename T>
HOSTDEVICE inline bool CalculateIoU(const T* const box_1,
                                    const T* const box_2,
                                    const float threshold) {
  auto box_1_x0 = box_1[0], box_1_y0 = box_1[1];
  auto box_1_x1 = box_1[2], box_1_y1 = box_1[3];
  auto box_2_x0 = box_2[0], box_2_y0 = box_2[1];
  auto box_2_x1 = box_2[2], box_2_y1 = box_2[3];

  auto inter_box_x0 = box_1_x0 > box_2_x0 ? box_1_x0 : box_2_x0;
  auto inter_box_y0 = box_1_y0 > box_2_y0 ? box_1_y0 : box_2_y0;
  auto inter_box_x1 = box_1_x1 < box_2_x1 ? box_1_x1 : box_2_x1;
  auto inter_box_y1 = box_1_y1 < box_2_y1 ? box_1_y1 : box_2_y1;

  auto inter_width =
      inter_box_x1 - inter_box_x0 > 0 ? inter_box_x1 - inter_box_x0 : 0;
  auto inter_height =
      inter_box_y1 - inter_box_y0 > 0 ? inter_box_y1 - inter_box_y0 : 0;
  auto inter_area = inter_width * inter_height;
  auto union_area = (box_1_x1 - box_1_x0) * (box_1_y1 - box_1_y0) +
                    (box_2_x1 - box_2_x0) * (box_2_y1 - box_2_y0) - inter_area;
  return inter_area / union_area > threshold;
}

template <typename T, typename Context>
void NMSKernel(const Context& dev_ctx,
               const DenseTensor& boxes,
               float threshold,
               DenseTensor* output);

}  // namespace phi
