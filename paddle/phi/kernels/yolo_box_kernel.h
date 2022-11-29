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
                   DenseTensor* scores);

}  // namespace phi
