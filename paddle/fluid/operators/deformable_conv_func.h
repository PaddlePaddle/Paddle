// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
//
// Part of the following code in this file refs to
// https://github.com/msracver/Deformable-ConvNets/blob/master/faster_rcnn/operator_cxx/deformable_convolution.cu
//
// Copyright (c) 2017 Microsoft
// Licensed under The Apache-2.0 License [see LICENSE for details]
// \file deformable_psroi_pooling.cu
// \brief
// \author Yi Li, Guodong Zhang, Jifeng Dai

#pragma once
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/pten/core/hostdevice.h"

template <typename T>
HOSTDEVICE T DmcnGetGradientWeight(T argmax_h, T argmax_w, const int h,
                                   const int w, const int height,
                                   const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  weight = (h == argmax_h_low && w == argmax_w_low)
               ? (h + 1 - argmax_h) * (w + 1 - argmax_w)
               : weight;
  weight = (h == argmax_h_low && w == argmax_w_high)
               ? (h + 1 - argmax_h) * (argmax_w + 1 - w)
               : weight;
  weight = (h == argmax_h_high && w == argmax_w_low)
               ? (argmax_h + 1 - h) * (w + 1 - argmax_w)
               : weight;
  weight = (h == argmax_h_high && w == argmax_w_high)
               ? (argmax_h + 1 - h) * (argmax_w + 1 - w)
               : weight;

  return weight;
}

template <typename T>
HOSTDEVICE T DmcnGetCoordinateWeight(T argmax_h, T argmax_w, const int height,
                                     const int width, const T* im_data,
                                     const int data_width, const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    weight += (argmax_h_low >= 0 && argmax_w_low >= 0)
                  ? -1 * (argmax_w_low + 1 - argmax_w) *
                        im_data[argmax_h_low * data_width + argmax_w_low]
                  : 0;

    weight += (argmax_h_low >= 0 && argmax_w_high <= width - 1)
                  ? -1 * (argmax_w - argmax_w_low) *
                        im_data[argmax_h_low * data_width + argmax_w_high]
                  : 0;

    weight += (argmax_h_high <= height - 1 && argmax_w_low >= 0)
                  ? (argmax_w_low + 1 - argmax_w) *
                        im_data[argmax_h_high * data_width + argmax_w_low]
                  : 0;
    weight += (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
                  ? (argmax_w - argmax_w_low) *
                        im_data[argmax_h_high * data_width + argmax_w_high]
                  : 0;
  } else if (bp_dir == 1) {
    weight += (argmax_h_low >= 0 && argmax_w_low >= 0)
                  ? -1 * (argmax_h_low + 1 - argmax_h) *
                        im_data[argmax_h_low * data_width + argmax_w_low]
                  : 0;
    weight += (argmax_h_low >= 0 && argmax_w_high <= width - 1)
                  ? (argmax_h_low + 1 - argmax_h) *
                        im_data[argmax_h_low * data_width + argmax_w_high]
                  : 0;
    weight += (argmax_h_high <= height - 1 && argmax_w_low >= 0)
                  ? -1 * (argmax_h - argmax_h_low) *
                        im_data[argmax_h_high * data_width + argmax_w_low]
                  : 0;
    weight += (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
                  ? (argmax_h - argmax_h_low) *
                        im_data[argmax_h_high * data_width + argmax_w_high]
                  : 0;
  }

  return weight;
}

template <typename T>
HOSTDEVICE T DmcnIm2colBilinear(const T* bottom_data, const int data_width,
                                const int height, const int width, T h, T w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh;
  T hw = 1 - lw;

  T v1 =
      (h_low >= 0 && w_low >= 0) ? bottom_data[h_low * data_width + w_low] : 0;
  T v2 = (h_low >= 0 && w_high <= width - 1)
             ? bottom_data[h_low * data_width + w_high]
             : 0;
  T v3 = (h_high <= height - 1 && w_low >= 0)
             ? bottom_data[h_high * data_width + w_low]
             : 0;
  T v4 = (h_high <= height - 1 && w_high <= width - 1)
             ? bottom_data[h_high * data_width + w_high]
             : 0;

  T w1 = hh * hw;
  T w2 = hh * lw;
  T w3 = lh * hw;
  T w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}
