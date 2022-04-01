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
void Yolov3LossGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& gt_box,
                          const DenseTensor& gt_label,
                          paddle::optional<const DenseTensor&> gt_score,
                          const DenseTensor& loss_grad,
                          const DenseTensor& objectness_mask,
                          const DenseTensor& gt_match_mask,
                          const std::vector<int>& anchors,
                          const std::vector<int>& anchor_mask,
                          int class_num,
                          float ignore_thresh,
                          int downsample_ratio,
                          bool use_label_smooth,
                          float scale_x_Y,
                          DenseTensor* x_grad,
                          DenseTensor* gt_box_grad,
                          DenseTensor* gt_label_grad,
                          DenseTensor* gt_score_grad);

}  // namespace phi
