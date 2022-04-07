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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
void Unnormalize(const CPUContext& ctx,
                 DenseTensor* grid_slice,
                 const int max_val,  // height-1 or width-1
                 bool align_corners) {
  auto& place = *ctx.eigen_device();
  auto grid_slice_t = EigenTensor<T, 3>::From(*grid_slice);

  if (!align_corners) {
    auto factor = static_cast<T>((max_val + 1) * 0.5);
    grid_slice_t.device(place) =
        (grid_slice_t + static_cast<T>(1)) * factor - static_cast<T>(0.5);
  } else {
    auto factor = static_cast<T>(max_val * 0.5);
    grid_slice_t.device(place) = (grid_slice_t + static_cast<T>(1)) * factor;
  }
}

template <typename T>
inline bool IsInBound(T x, T y, T x_max, T y_max) {
  if (x < 0 || x > x_max || y < 0 || y > y_max) {
    return false;
  }
  return true;
}

template <typename T>
void GetGridPointValue(const DenseTensor& input,
                       DenseTensor* output,
                       const DenseTensor& x,
                       const DenseTensor& y) {
  const int n = input.dims()[0];
  const int c = input.dims()[1];
  const int in_h = input.dims()[2];
  const int in_w = input.dims()[3];
  const int out_h = x.dims()[1];
  const int out_w = x.dims()[2];
  auto x_t = EigenTensor<T, 3>::From(x);
  auto y_t = EigenTensor<T, 3>::From(y);
  auto output_t = EigenTensor<T, 4>::From(*output).setConstant((T)0);
  auto input_t = EigenTensor<T, 4>::From(input);

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < out_h; k++) {
      for (int l = 0; l < out_w; l++) {
        if (IsInBound(
                x_t(i, k, l), y_t(i, k, l), (T)(in_w - 1), (T)(in_h - 1))) {
          for (int j = 0; j < c; j++) {
            output_t(i, j, k, l) =
                input_t(i,
                        j,
                        static_cast<int>(round(y_t(i, k, l))),
                        static_cast<int>(round(x_t(i, k, l))));
          }
        }
      }
    }
  }
}

template <typename T>
void AllNeigbors(const CPUContext& ctx,
                 const DenseTensor& input,
                 DenseTensor* grid_x,
                 DenseTensor* grid_y,
                 DenseTensor* x_w,
                 DenseTensor* x_e,
                 DenseTensor* y_n,
                 DenseTensor* y_s,  // positions
                 DenseTensor* d_w,
                 DenseTensor* d_e,
                 DenseTensor* d_n,
                 DenseTensor* d_s,  // distance
                 DenseTensor* v_wn,
                 DenseTensor* v_en,
                 DenseTensor* v_ws,
                 DenseTensor* v_es) {  // values
  auto& place = *ctx.eigen_device();

  const int c = input.dims()[1];
  const int n = grid_x->dims()[0];
  const int out_h = grid_x->dims()[1];
  const int out_w = grid_x->dims()[2];
  // calculate coords of 4 corner points
  x_w->Resize({n, out_h, out_w});
  x_e->Resize({n, out_h, out_w});
  y_n->Resize({n, out_h, out_w});
  y_s->Resize({n, out_h, out_w});
  ctx.Alloc<T>(x_w);
  ctx.Alloc<T>(x_e);
  ctx.Alloc<T>(y_n);
  ctx.Alloc<T>(y_s);
  auto x_w_t = EigenTensor<T, 3>::From(*x_w);
  auto x_e_t = EigenTensor<T, 3>::From(*x_e);
  auto y_n_t = EigenTensor<T, 3>::From(*y_n);
  auto y_s_t = EigenTensor<T, 3>::From(*y_s);

  auto grid_x_t = EigenTensor<T, 3>::From(*grid_x);
  auto grid_y_t = EigenTensor<T, 3>::From(*grid_y);

  x_w_t.device(place) = grid_x_t.floor();
  x_e_t.device(place) = x_w_t + static_cast<T>(1);
  y_n_t.device(place) = grid_y_t.floor();
  y_s_t.device(place) = y_n_t + static_cast<T>(1);

  // calculate distances to 4 sides
  d_w->Resize({n, out_h, out_w});
  d_e->Resize({n, out_h, out_w});
  d_n->Resize({n, out_h, out_w});
  d_s->Resize({n, out_h, out_w});
  ctx.Alloc<T>(d_w);
  ctx.Alloc<T>(d_e);
  ctx.Alloc<T>(d_n);
  ctx.Alloc<T>(d_s);
  auto d_w_t = EigenTensor<T, 3>::From(*d_w);
  auto d_e_t = EigenTensor<T, 3>::From(*d_e);
  auto d_n_t = EigenTensor<T, 3>::From(*d_n);
  auto d_s_t = EigenTensor<T, 3>::From(*d_s);
  d_w_t.device(place) = grid_x_t - x_w_t;
  d_e_t.device(place) = x_e_t - grid_x_t;
  d_n_t.device(place) = grid_y_t - y_n_t;
  d_s_t.device(place) = y_s_t - grid_y_t;

  // calc 4 corner points value
  v_wn->Resize({n, c, out_h, out_w});
  v_en->Resize({n, c, out_h, out_w});
  v_ws->Resize({n, c, out_h, out_w});
  v_es->Resize({n, c, out_h, out_w});
  ctx.Alloc<T>(v_wn);
  ctx.Alloc<T>(v_en);
  ctx.Alloc<T>(v_ws);
  ctx.Alloc<T>(v_es);
  GetGridPointValue<T>(input, v_wn, *x_w, *y_n);
  GetGridPointValue<T>(input, v_en, *x_e, *y_n);
  GetGridPointValue<T>(input, v_ws, *x_w, *y_s);
  GetGridPointValue<T>(input, v_es, *x_e, *y_s);
}

}  // namespace phi
