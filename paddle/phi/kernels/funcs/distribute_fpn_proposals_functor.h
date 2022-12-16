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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

const int kBoxDim = 4;

template <typename Context>
inline std::vector<size_t> GetLodFromRoisNum(const Context& dev_ctx,
                                             const DenseTensor* rois_num) {
  std::vector<size_t> rois_lod;
  auto* rois_num_data = rois_num->data<int>();
  DenseTensor cpu_tensor;
  if (paddle::platform::is_gpu_place(rois_num->place())) {
    Copy<Context>(dev_ctx, *rois_num, phi::CPUPlace(), true, &cpu_tensor);
    rois_num_data = cpu_tensor.data<int>();
  }
  rois_lod.push_back(static_cast<size_t>(0));
  for (int i = 0; i < rois_num->numel(); ++i) {
    rois_lod.push_back(rois_lod.back() + static_cast<size_t>(rois_num_data[i]));
  }
  return rois_lod;
}

template <typename T>
static inline T BBoxArea(const T* box, bool pixel_offset) {
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

}  // namespace funcs
}  // namespace phi
