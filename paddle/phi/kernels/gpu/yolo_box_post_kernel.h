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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

struct Box {
  float x, y, w, h;
};

struct Detection {
  Box bbox;
  int classes;
  float* prob;
  float* mask;
  float objectness;
  int sort_class;
  int max_prob_class_index;
};

struct TensorInfo {
  int bbox_count_host;  // record bbox numbers
  int bbox_count_max_alloc{50};
  float* bboxes_dev_ptr;
  float* bboxes_host_ptr;
  int* bbox_count_device_ptr;  // Box counter in gpu memory, used by atomicAdd
};

template <typename T, typename Context>
void YoloBoxPostKernel(const Context& dev_ctx,
                       const DenseTensor& boxes0,
                       const DenseTensor& boxes1,
                       const DenseTensor& boxes2,
                       const DenseTensor& image_shape,
                       const DenseTensor& image_scale,
                       const std::vector<int>& anchors0,
                       const std::vector<int>& anchors1,
                       const std::vector<int>& anchors2,
                       int class_num,
                       float conf_thresh,
                       int downsample_ratio0,
                       int downsample_ratio1,
                       int downsample_ratio2,
                       bool clip_bbox,
                       float scale_x_y,
                       float nms_threshold,
                       DenseTensor* out,
                       DenseTensor* nms_rois_num);

}  // namespace phi
