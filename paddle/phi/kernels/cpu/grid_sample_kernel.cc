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
using Array5 = Eigen::DSizes<int64_t, 5>;

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
static inline void Clip3D(const CPUContext& ctx,
                          DenseTensor* grid_slice,
                          const int max_val,  // height-1 or width-1
                          bool align_corners,
                          std::string padding_mode) {
  auto& place = *ctx.eigen_device();
  auto grid_slice_t = EigenTensor<T, 4>::From(*grid_slice);
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
static void Calc3DGridLocations(const CPUContext& ctx,
                                const DenseTensor& grid,
                                const int in_d,
                                const int in_h,
                                const int in_w,
                                bool align_corners,
                                std::string padding_mode,
                                DenseTensor* grid_x,
                                DenseTensor* grid_y,
                                DenseTensor* grid_z) {
  const int n = grid.dims()[0];
  const int out_d = grid.dims()[1];
  const int out_h = grid.dims()[2];
  const int out_w = grid.dims()[3];

  // split grid with shape (n, d, h, w, 3) into (x, y, z) by the 3rd Dim
  grid_x->Resize({n, out_d, out_h, out_w});
  grid_y->Resize({n, out_d, out_h, out_w});
  grid_z->Resize({n, out_d, out_h, out_w});
  T* grid_x_data = ctx.Alloc<T>(grid_x);
  T* grid_y_data = ctx.Alloc<T>(grid_y);
  T* grid_z_data = ctx.Alloc<T>(grid_z);
  const T* grid_data = grid.data<T>();
  for (int i = 0; i < n * out_d * out_h * out_w; i++) {
    grid_x_data[i] = grid_data[3 * i];
    grid_y_data[i] = grid_data[(3 * i) + 1];
    grid_z_data[i] = grid_data[(3 * i) + 2];
  }

  Unnormalize3D<T>(ctx, grid_x, in_w - 1, align_corners);
  Unnormalize3D<T>(ctx, grid_y, in_h - 1, align_corners);
  Unnormalize3D<T>(ctx, grid_z, in_d - 1, align_corners);

  Clip3D<T>(ctx, grid_x, in_w - 1, align_corners, padding_mode);
  Clip3D<T>(ctx, grid_y, in_h - 1, align_corners, padding_mode);
  Clip3D<T>(ctx, grid_z, in_d - 1, align_corners, padding_mode);
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

template <typename T>
static void Bilinear3DInter(const CPUContext& ctx,
                            const DenseTensor& input,
                            DenseTensor* grid_x,
                            DenseTensor* grid_y,
                            DenseTensor* grid_z,
                            DenseTensor* out) {
  auto& place = *ctx.eigen_device();
  const int n = grid_x->dims()[0];
  const int out_d = grid_x->dims()[1];
  const int out_h = grid_x->dims()[2];
  const int out_w = grid_x->dims()[3];
  const int c = input.dims()[1];

  // get corner pixel values from (x, y, z)
  // for 4d, we used north-east-south-west
  // for 5d, we add top-bottom
  DenseTensor x_w, x_e, y_n, y_s, z_t, z_b;
  DenseTensor d_w, d_e, d_n, d_s, d_t, d_b;
  DenseTensor v_twn, v_ten, v_tws, v_tes, v_bwn, v_ben, v_bws, v_bes;

  All3DNeigbors<T>(ctx,
                   input,
                   grid_x,
                   grid_y,
                   grid_z,
                   &x_w,
                   &x_e,
                   &y_n,
                   &y_s,
                   &z_t,
                   &z_b,
                   &d_w,
                   &d_e,
                   &d_n,
                   &d_s,
                   &d_t,
                   &d_b,
                   &v_twn,
                   &v_ten,
                   &v_tws,
                   &v_tes,
                   &v_bwn,
                   &v_ben,
                   &v_bws,
                   &v_bes);

  auto d_w_t = EigenTensor<T, 4>::From(d_w);
  auto d_e_t = EigenTensor<T, 4>::From(d_e);
  auto d_n_t = EigenTensor<T, 4>::From(d_n);
  auto d_s_t = EigenTensor<T, 4>::From(d_s);
  auto d_t_t = EigenTensor<T, 4>::From(d_t);
  auto d_b_t = EigenTensor<T, 4>::From(d_b);

  auto d_w_scaled_t = d_w_t.reshape(Array5(n, 1, out_d, out_h, out_w))
                          .broadcast(Array5(1, c, 1, 1, 1));
  auto d_e_scaled_t = d_e_t.reshape(Array5(n, 1, out_d, out_h, out_w))
                          .broadcast(Array5(1, c, 1, 1, 1));
  auto d_n_scaled_t = d_n_t.reshape(Array5(n, 1, out_d, out_h, out_w))
                          .broadcast(Array5(1, c, 1, 1, 1));
  auto d_s_scaled_t = d_s_t.reshape(Array5(n, 1, out_d, out_h, out_w))
                          .broadcast(Array5(1, c, 1, 1, 1));
  auto d_t_scaled_t = d_t_t.reshape(Array5(n, 1, out_d, out_h, out_w))
                          .broadcast(Array5(1, c, 1, 1, 1));
  auto d_b_scaled_t = d_b_t.reshape(Array5(n, 1, out_d, out_h, out_w))
                          .broadcast(Array5(1, c, 1, 1, 1));

  auto v_twn_t = EigenTensor<T, 5>::From(v_twn);
  auto v_ten_t = EigenTensor<T, 5>::From(v_ten);
  auto v_tws_t = EigenTensor<T, 5>::From(v_tws);
  auto v_tes_t = EigenTensor<T, 5>::From(v_tes);
  auto v_bwn_t = EigenTensor<T, 5>::From(v_bwn);
  auto v_ben_t = EigenTensor<T, 5>::From(v_ben);
  auto v_bws_t = EigenTensor<T, 5>::From(v_bws);
  auto v_bes_t = EigenTensor<T, 5>::From(v_bes);
  auto output_t = EigenTensor<T, 5>::From(*out);
  // bilinear interpolaetion by 4 corner points
  output_t.device(place) =
      v_twn_t * d_e_scaled_t * d_s_scaled_t * d_b_scaled_t +
      v_ten_t * d_w_scaled_t * d_s_scaled_t * d_b_scaled_t +
      v_tws_t * d_e_scaled_t * d_n_scaled_t * d_b_scaled_t +
      v_tes_t * d_w_scaled_t * d_n_scaled_t * d_b_scaled_t +
      v_bwn_t * d_e_scaled_t * d_s_scaled_t * d_t_scaled_t +
      v_ben_t * d_w_scaled_t * d_s_scaled_t * d_t_scaled_t +
      v_bws_t * d_e_scaled_t * d_n_scaled_t * d_t_scaled_t +
      v_bes_t * d_w_scaled_t * d_n_scaled_t * d_t_scaled_t;
}

template <typename T, typename Context>
void GridSampleKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& grid,
                      const std::string& mode,
                      const std::string& padding_mode,
                      bool align_corners,
                      DenseTensor* out) {
  if (x.dims().size() == 4) {
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
    CalcGridLocations<T>(dev_ctx,
                         grid,
                         in_h,
                         in_w,
                         align_corners,
                         padding_mode,
                         &grid_x,
                         &grid_y);

    if (mode == "bilinear") {
      BilinearInter<T>(dev_ctx, x, &grid_x, &grid_y, out);
    } else if (mode == "nearest") {
      auto grid_x_t = EigenTensor<T, 3>::From(grid_x);
      auto grid_y_t = EigenTensor<T, 3>::From(grid_y);
      grid_x_t = grid_x_t.round();
      grid_y_t = grid_y_t.round();
      GetGridPointValue<T>(x, out, grid_x, grid_y);
    }
  } else {
    const int n = grid.dims()[0];
    const int out_d = grid.dims()[1];
    const int out_h = grid.dims()[2];
    const int out_w = grid.dims()[3];
    const int c = x.dims()[1];
    const int in_d = x.dims()[2];
    const int in_h = x.dims()[3];
    const int in_w = x.dims()[4];

    out->Resize(phi::make_ddim({n, c, out_d, out_h, out_w}));
    dev_ctx.template Alloc<T>(out);
    phi::funcs::SetConstant<Context, T>()(dev_ctx, out, static_cast<T>(0));

    DenseTensor grid_x, grid_y, grid_z;
    Calc3DGridLocations<T>(dev_ctx,
                           grid,
                           in_d,
                           in_h,
                           in_w,
                           align_corners,
                           padding_mode,
                           &grid_x,
                           &grid_y,
                           &grid_z);
    if (mode == "bilinear") {
      Bilinear3DInter<T>(dev_ctx, x, &grid_x, &grid_y, &grid_z, out);
    } else if (mode == "nearest") {
      Get3DGridPointValue<T>(x, out, grid_x, grid_y, grid_z);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    grid_sample, CPU, ALL_LAYOUT, phi::GridSampleKernel, float, double) {}
