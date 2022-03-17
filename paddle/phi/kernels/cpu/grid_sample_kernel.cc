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

#include "paddle/phi/kernels/grid_sample_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/grid_sample_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

using Array4 = Eigen::DSizes<int64_t, 4>;

template <typename T>
static inline void Clip(const CPUContext& ctx,
                        DenseTensor* grid_slice,
                        const int max_val,  // height-1 or width-1
                        bool align_corners,
                        std::string padding_mode) {
  auto& place = *ctx.eigen_device();
  auto grid_slice_t = EigenTensor<T, 3>::From(*grid_slice);
  if (padding_mode == "border") {
    grid_slice_t.device(place) = grid_slice_t.cwiseMax(static_cast<T>(0))
                                     .cwiseMin(static_cast<T>(max_val));
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      auto double_range = static_cast<T>(max_val * 2);
      auto grid_abs = grid_slice_t.abs();
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      grid_slice_t.device(place) = extra.cwiseMin(double_range - extra);
      if (max_val == 0) {
        grid_slice_t.device(place) = grid_slice_t.constant(static_cast<T>(0));
      }
    } else {
      auto double_range = static_cast<T>((max_val + 1) * 2);
      auto grid_abs = (grid_slice_t + static_cast<T>(0.5)).abs();
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      grid_slice_t.device(place) =
          extra.cwiseMin(double_range - extra) - static_cast<T>(0.5);
      grid_slice_t.device(place) = grid_slice_t.cwiseMax(static_cast<T>(0))
                                       .cwiseMin(static_cast<T>(max_val));
    }
  }
}

template <typename T>
static void CalcGridLocations(const CPUContext& ctx,
                              const DenseTensor& grid,
                              const int in_h,
                              const int in_w,
                              bool align_corners,
                              std::string padding_mode,
                              DenseTensor* grid_x,
                              DenseTensor* grid_y) {
  const int n = grid.dims()[0];
  const int out_h = grid.dims()[1];
  const int out_w = grid.dims()[2];

  // split grid with shape (n, h, w, 2) into (x, y) by the 3rd Dim
  grid_x->Resize({n, out_h, out_w});
  grid_y->Resize({n, out_h, out_w});
  T* grid_x_data = ctx.Alloc<T>(grid_x);
  T* grid_y_data = ctx.Alloc<T>(grid_y);
  const T* grid_data = grid.data<T>();
  for (int i = 0; i < n * out_h * out_w; i++) {
    grid_x_data[i] = grid_data[2 * i];
    grid_y_data[i] = grid_data[(2 * i) + 1];
  }

  Unnormalize<T>(ctx, grid_x, in_w - 1, align_corners);
  Unnormalize<T>(ctx, grid_y, in_h - 1, align_corners);

  Clip<T>(ctx, grid_x, in_w - 1, align_corners, padding_mode);
  Clip<T>(ctx, grid_y, in_h - 1, align_corners, padding_mode);
}

template <typename T>
static void BilinearInter(const CPUContext& ctx,
                          const DenseTensor& input,
                          DenseTensor* grid_x,
                          DenseTensor* grid_y,
                          DenseTensor* out) {
  auto& place = *ctx.eigen_device();
  const int n = grid_x->dims()[0];
  const int out_h = grid_x->dims()[1];
  const int out_w = grid_x->dims()[2];
  const int c = input.dims()[1];

  DenseTensor x_w, x_e, y_n, y_s;
  DenseTensor d_w, d_e, d_n, d_s;
  DenseTensor v_wn, v_en, v_ws, v_es;

  AllNeigbors<T>(ctx,
                 input,
                 grid_x,
                 grid_y,
                 &x_w,
                 &x_e,
                 &y_n,
                 &y_s,
                 &d_w,
                 &d_e,
                 &d_n,
                 &d_s,
                 &v_wn,
                 &v_en,
                 &v_ws,
                 &v_es);

  auto d_w_t = EigenTensor<T, 3>::From(d_w);
  auto d_e_t = EigenTensor<T, 3>::From(d_e);
  auto d_n_t = EigenTensor<T, 3>::From(d_n);
  auto d_s_t = EigenTensor<T, 3>::From(d_s);

  auto d_w_scaled_t =
      d_w_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto d_e_scaled_t =
      d_e_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto d_n_scaled_t =
      d_n_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto d_s_scaled_t =
      d_s_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto v_wn_t = EigenTensor<T, 4>::From(v_wn);
  auto v_en_t = EigenTensor<T, 4>::From(v_en);
  auto v_ws_t = EigenTensor<T, 4>::From(v_ws);
  auto v_es_t = EigenTensor<T, 4>::From(v_es);
  auto output_t = EigenTensor<T, 4>::From(*out);
  // bilinear interpolaetion by 4 corner points
  output_t.device(place) = v_wn_t * d_e_scaled_t * d_s_scaled_t +
                           v_en_t * d_w_scaled_t * d_s_scaled_t +
                           v_ws_t * d_e_scaled_t * d_n_scaled_t +
                           v_es_t * d_w_scaled_t * d_n_scaled_t;
}

template <typename T, typename Context>
void GridSampleKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& grid,
                      const std::string& mode,
                      const std::string& padding_mode,
                      bool align_corners,
                      DenseTensor* out) {
  const int n = grid.dims()[0];
  const int out_h = grid.dims()[1];
  const int out_w = grid.dims()[2];
  const int c = x.dims()[1];
  const int in_h = x.dims()[2];
  const int in_w = x.dims()[3];

  out->Resize(phi::make_ddim({n, c, out_h, out_w}));
  dev_ctx.template Alloc<T>(out);
  phi::funcs::SetConstant<Context, T>()(dev_ctx, out, static_cast<T>(0));

  DenseTensor grid_x, grid_y;
  CalcGridLocations<T>(
      dev_ctx, grid, in_h, in_w, align_corners, padding_mode, &grid_x, &grid_y);

  if (mode == "bilinear") {
    BilinearInter<T>(dev_ctx, x, &grid_x, &grid_y, out);
  } else if (mode == "nearest") {
    auto grid_x_t = EigenTensor<T, 3>::From(grid_x);
    auto grid_y_t = EigenTensor<T, 3>::From(grid_y);
    grid_x_t = grid_x_t.round();
    grid_y_t = grid_y_t.round();
    GetGridPointValue<T>(x, out, grid_x, grid_y);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    grid_sample, CPU, ALL_LAYOUT, phi::GridSampleKernel, float, double) {}
