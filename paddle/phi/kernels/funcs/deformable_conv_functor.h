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

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace funcs {

template <typename T, typename MT>
HOSTDEVICE MT DmcnIm2colBilinear(const T* bottom_data,
                                 const int data_width,
                                 const int height,
                                 const int width,
                                 MT h,
                                 MT w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  MT lh = h - h_low;
  MT lw = w - w_low;
  MT hh = 1 - lh;
  MT hw = 1 - lw;

  MT v1 = (h_low >= 0 && w_low >= 0)
              ? static_cast<MT>(bottom_data[h_low * data_width + w_low])
              : 0;
  MT v2 = (h_low >= 0 && w_high <= width - 1)
              ? static_cast<MT>(bottom_data[h_low * data_width + w_high])
              : 0;
  MT v3 = (h_high <= height - 1 && w_low >= 0)
              ? static_cast<MT>(bottom_data[h_high * data_width + w_low])
              : 0;
  MT v4 = (h_high <= height - 1 && w_high <= width - 1)
              ? static_cast<MT>(bottom_data[h_high * data_width + w_high])
              : 0;

  MT w1 = hh * hw;
  MT w2 = hh * lw;
  MT w3 = lh * hw;
  MT w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename T, typename Context>
void ModulatedDeformableIm2col(const Context& dev_ctx,
                               const T* data_im,
                               const T* data_offset,
                               const T* data_mask,
                               const std::vector<int64_t>& im_shape,
                               const std::vector<int64_t>& col_shape,
                               const std::vector<int64_t>& filter_shape,
                               const std::vector<int>& paddings,
                               const std::vector<int>& strides,
                               const std::vector<int>& dilations,
                               const int deformable_groups,
                               T* data_col);

}  // namespace funcs
}  // namespace phi
