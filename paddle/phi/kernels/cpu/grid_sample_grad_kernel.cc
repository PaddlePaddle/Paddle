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

#include "paddle/phi/kernels/grid_sample_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/grid_sample_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
static inline void ClipWithMask(const CPUContext& ctx,
                                const int max_val,  // height-1 or width-1
                                bool align_corners,
                                std::string padding_mode,
                                DenseTensor* grid_slice,
                                DenseTensor* grid_scale) {
  auto& place = *ctx.eigen_device();
  grid_scale->Resize(grid_slice->dims());
  ctx.Alloc<T>(grid_scale);

  auto grid_slice_t = EigenTensor<T, 3>::From(*grid_slice);
  auto factor = static_cast<T>(max_val * 0.5);
  if (!align_corners) {
    factor = static_cast<T>((max_val + 1) * 0.5);
  }
  auto grid_scale_t = EigenTensor<T, 3>::From(*grid_scale).setConstant(factor);

  if (padding_mode == "border") {
    //    auto bounded_lo = grid_slice_t.cwiseMax(static_cast<T>(0));
    auto res = grid_slice_t.cwiseMax(static_cast<T>(0))
                   .cwiseMin(static_cast<T>(max_val));

    auto in_bound = (res == grid_slice_t);
    grid_scale_t.device(place) = grid_scale_t * in_bound.template cast<T>();
    grid_slice_t.device(place) = res;
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      auto double_range = static_cast<T>(max_val * 2);
      auto is_neg = (grid_slice_t < static_cast<T>(0));
      auto grid_abs = grid_slice_t.abs();
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      auto one_more_flip = (extra > (double_range - extra));
      grid_scale_t.device(place) =
          grid_scale_t * ((is_neg == one_more_flip).template cast<T>() -
                          (is_neg != one_more_flip).template cast<T>());
      grid_slice_t.device(place) = extra.cwiseMin(double_range - extra);
      if (max_val == 0) {
        grid_slice_t.device(place) = grid_slice_t.constant(static_cast<T>(0));
      }
    } else {
      auto double_range = static_cast<T>((max_val + 1) * 2);
      auto grid_abs = (grid_slice_t + static_cast<T>(0.5)).abs();
      auto is_neg = ((grid_slice_t + static_cast<T>(0.5)) < static_cast<T>(0));
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      auto one_more_flip = (extra > (double_range - extra));
      auto reflected =
          extra.cwiseMin(double_range - extra) - static_cast<T>(0.5);
      auto clipped = reflected.cwiseMax(static_cast<T>(0))
                         .cwiseMin(static_cast<T>(max_val));
      auto in_bound = (clipped == reflected).template cast<T>();
      grid_scale_t.device(place) =
          grid_scale_t *
          ((is_neg == one_more_flip).template cast<T>() -
           (is_neg != one_more_flip).template cast<T>()) *
          in_bound;
      grid_slice_t.device(place) = clipped;
    }
  }
}

template <typename T>
static inline void ClipWithMask3D(const CPUContext& ctx,
                                  const int max_val,  // height-1 or width-1
                                  bool align_corners,
                                  std::string padding_mode,
                                  DenseTensor* grid_slice,
                                  DenseTensor* grid_scale) {
  auto& place = *ctx.eigen_device();
  grid_scale->Resize(grid_slice->dims());
  ctx.Alloc<T>(grid_scale);

  auto grid_slice_t = EigenTensor<T, 4>::From(*grid_slice);
  auto factor = static_cast<T>(max_val * 0.5);
  if (!align_corners) {
    factor = static_cast<T>((max_val + 1) * 0.5);
  }
  auto grid_scale_t = EigenTensor<T, 4>::From(*grid_scale).setConstant(factor);

  if (padding_mode == "border") {
    //    auto bounded_lo = grid_slice_t.cwiseMax(static_cast<T>(0));
    auto res = grid_slice_t.cwiseMax(static_cast<T>(0))
                   .cwiseMin(static_cast<T>(max_val));

    auto in_bound = (res == grid_slice_t);
    grid_scale_t.device(place) = grid_scale_t * in_bound.template cast<T>();
    grid_slice_t.device(place) = res;
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      auto double_range = static_cast<T>(max_val * 2);
      auto is_neg = (grid_slice_t < static_cast<T>(0));
      auto grid_abs = grid_slice_t.abs();
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      auto one_more_flip = (extra > (double_range - extra));
      grid_scale_t.device(place) =
          grid_scale_t * ((is_neg == one_more_flip).template cast<T>() -
                          (is_neg != one_more_flip).template cast<T>());
      grid_slice_t.device(place) = extra.cwiseMin(double_range - extra);
      if (max_val == 0) {
        grid_slice_t.device(place) = grid_slice_t.constant(static_cast<T>(0));
      }
    } else {
      auto double_range = static_cast<T>((max_val + 1) * 2);
      auto grid_abs = (grid_slice_t + static_cast<T>(0.5)).abs();
      auto is_neg = ((grid_slice_t + static_cast<T>(0.5)) < static_cast<T>(0));
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      auto one_more_flip = (extra > (double_range - extra));
      auto reflected =
          extra.cwiseMin(double_range - extra) - static_cast<T>(0.5);
      auto clipped = reflected.cwiseMax(static_cast<T>(0))
                         .cwiseMin(static_cast<T>(max_val));
      auto in_bound = (clipped == reflected).template cast<T>();
      grid_scale_t.device(place) =
          grid_scale_t *
          ((is_neg == one_more_flip).template cast<T>() -
           (is_neg != one_more_flip).template cast<T>()) *
          in_bound;
      grid_slice_t.device(place) = clipped;
    }
  }
}

template <typename T>
static void CalcGridLocationsWithGrad(const CPUContext& ctx,
                                      const DenseTensor& grid,
                                      const int in_h,
                                      const int in_w,
                                      bool align_corners,
                                      std::string padding_mode,
                                      DenseTensor* grid_x,
                                      DenseTensor* grid_y,
                                      DenseTensor* grid_x_scale,
                                      DenseTensor* grid_y_scale) {
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

  ClipWithMask<T>(
      ctx, in_w - 1, align_corners, padding_mode, grid_x, grid_x_scale);
  ClipWithMask<T>(
      ctx, in_h - 1, align_corners, padding_mode, grid_y, grid_y_scale);
}

template <typename T>
static void Calc3DGridLocationsWithGrad(const CPUContext& ctx,
                                        const DenseTensor& grid,
                                        const int in_d,
                                        const int in_h,
                                        const int in_w,
                                        bool align_corners,
                                        std::string padding_mode,
                                        DenseTensor* grid_x,
                                        DenseTensor* grid_y,
                                        DenseTensor* grid_z,
                                        DenseTensor* grid_x_scale,
                                        DenseTensor* grid_y_scale,
                                        DenseTensor* grid_z_scale) {
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

  ClipWithMask3D<T>(
      ctx, in_w - 1, align_corners, padding_mode, grid_x, grid_x_scale);
  ClipWithMask3D<T>(
      ctx, in_h - 1, align_corners, padding_mode, grid_y, grid_y_scale);
  ClipWithMask3D<T>(
      ctx, in_d - 1, align_corners, padding_mode, grid_z, grid_z_scale);
}

template <typename T>
static void GatherOutputGradToInputGrad(const DenseTensor& output_grad,
                                        DenseTensor* input_grad,
                                        const DenseTensor& x,
                                        const DenseTensor& y,
                                        const DenseTensor& d1,
                                        const DenseTensor& d2) {
  const int n = output_grad.dims()[0];
  const int c = output_grad.dims()[1];
  const int out_h = output_grad.dims()[2];
  const int out_w = output_grad.dims()[3];
  const int in_h = input_grad->dims()[2];
  const int in_w = input_grad->dims()[3];
  auto x_t = EigenTensor<T, 3>::From(x);
  auto y_t = EigenTensor<T, 3>::From(y);
  auto d1_t = EigenTensor<T, 3>::From(d1);
  auto d2_t = EigenTensor<T, 3>::From(d2);
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < out_h; k++) {
      for (int l = 0; l < out_w; l++) {
        if (IsInBound(
                x_t(i, k, l), y_t(i, k, l), (T)(in_w - 1), (T)(in_h - 1))) {
          for (int j = 0; j < c; j++) {
            input_grad_t(i,
                         j,
                         static_cast<int>(round(y_t(i, k, l))),
                         static_cast<int>(round(x_t(i, k, l)))) +=
                output_grad_t(i, j, k, l) * d1_t(i, k, l) * d2_t(i, k, l);
          }
        }
      }
    }
  }
}

template <typename T>
static void Gather3DOutputGradToInputGrad(const DenseTensor& output_grad,
                                          DenseTensor* input_grad,
                                          const DenseTensor& x,
                                          const DenseTensor& y,
                                          const DenseTensor& z,
                                          const DenseTensor& d1,
                                          const DenseTensor& d2,
                                          const DenseTensor& d3) {
  const int n = output_grad.dims()[0];
  const int c = output_grad.dims()[1];
  const int out_d = output_grad.dims()[2];
  const int out_h = output_grad.dims()[3];
  const int out_w = output_grad.dims()[4];
  const int in_d = input_grad->dims()[2];
  const int in_h = input_grad->dims()[3];
  const int in_w = input_grad->dims()[4];
  auto x_t = EigenTensor<T, 4>::From(x);
  auto y_t = EigenTensor<T, 4>::From(y);
  auto z_t = EigenTensor<T, 4>::From(z);
  auto d1_t = EigenTensor<T, 4>::From(d1);
  auto d2_t = EigenTensor<T, 4>::From(d2);
  auto d3_t = EigenTensor<T, 4>::From(d3);
  auto input_grad_t = EigenTensor<T, 5>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 5>::From(output_grad);

  for (int i = 0; i < n; i++) {
    for (int m = 0; m < out_d; m++) {
      for (int k = 0; k < out_h; k++) {
        for (int l = 0; l < out_w; l++) {
          if (IsInBound3D(x_t(i, m, k, l),
                          y_t(i, m, k, l),
                          z_t(i, m, k, l),
                          (T)(in_w - 1),
                          (T)(in_h - 1),
                          (T)(in_d - 1))) {
            for (int j = 0; j < c; j++) {
              input_grad_t(i,
                           j,
                           static_cast<int>(round(z_t(i, m, k, l))),
                           static_cast<int>(round(y_t(i, m, k, l))),
                           static_cast<int>(round(x_t(i, m, k, l)))) +=
                  output_grad_t(i, j, m, k, l) * d1_t(i, m, k, l) *
                  d2_t(i, m, k, l) * d3_t(i, m, k, l);
            }
          }
        }
      }
    }
  }
}

template <typename T>
static void GatherBilinearGrad(const CPUContext& ctx,
                               const DenseTensor& input,
                               const DenseTensor& output_grad,
                               DenseTensor* grid_x,
                               DenseTensor* grid_y,
                               DenseTensor* grid_x_scale,
                               DenseTensor* grid_y_scale,
                               DenseTensor* input_grad,
                               DenseTensor* grid_grad) {
  const int n = grid_x->dims()[0];
  const int out_h = grid_x->dims()[1];
  const int out_w = grid_x->dims()[2];
  const int c = input.dims()[1];

  DenseTensor x_w, x_e, y_n, y_s;
  DenseTensor d_w, d_e, d_n, d_s;
  DenseTensor v_wn, v_en, v_ws, v_es;

  AllNeigbors<T>(ctx,
                 input,
                 grid_x,  // grid_x
                 grid_y,  // grid_y
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

  // gather output grad value to input grad by corner point coords and weight
  GatherOutputGradToInputGrad<T>(output_grad, input_grad, x_w, y_n, d_e, d_s);
  GatherOutputGradToInputGrad<T>(output_grad, input_grad, x_w, y_s, d_e, d_n);
  GatherOutputGradToInputGrad<T>(output_grad, input_grad, x_e, y_n, d_w, d_s);
  GatherOutputGradToInputGrad<T>(output_grad, input_grad, x_e, y_s, d_w, d_n);

  auto v_wn_t = EigenTensor<T, 4>::From(v_wn);
  auto v_en_t = EigenTensor<T, 4>::From(v_en);
  auto v_ws_t = EigenTensor<T, 4>::From(v_ws);
  auto v_es_t = EigenTensor<T, 4>::From(v_es);

  auto d_w_t = EigenTensor<T, 3>::From(d_w);
  auto d_e_t = EigenTensor<T, 3>::From(d_e);
  auto d_n_t = EigenTensor<T, 3>::From(d_n);
  auto d_s_t = EigenTensor<T, 3>::From(d_s);

  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);

  if (grid_grad != nullptr) {
    DenseTensor grid_grad_x, grid_grad_y;
    grid_grad_x.Resize({n, out_h, out_w});
    grid_grad_y.Resize({n, out_h, out_w});
    ctx.Alloc<T>(&grid_grad_x);
    ctx.Alloc<T>(&grid_grad_y);
    auto grid_grad_x_t =
        EigenTensor<T, 3>::From(grid_grad_x).setConstant(static_cast<T>(0.0));
    auto grid_grad_y_t =
        EigenTensor<T, 3>::From(grid_grad_y).setConstant(static_cast<T>(0.0));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < c; j++) {
        for (int k = 0; k < out_h; k++) {
          for (int l = 0; l < out_w; l++) {
            grid_grad_x_t(i, k, l) +=
                ((v_en_t(i, j, k, l) - v_wn_t(i, j, k, l)) * d_s_t(i, k, l) +
                 (v_es_t(i, j, k, l) - v_ws_t(i, j, k, l)) * d_n_t(i, k, l)) *
                output_grad_t(i, j, k, l);
            grid_grad_y_t(i, k, l) +=
                ((v_ws_t(i, j, k, l) - v_wn_t(i, j, k, l)) * d_e_t(i, k, l) +
                 (v_es_t(i, j, k, l) - v_en_t(i, j, k, l)) * d_w_t(i, k, l)) *
                output_grad_t(i, j, k, l);
          }
        }
      }
    }

    //  const T x_max = static_cast<T>(in_w - 1);
    //  const T y_max = static_cast<T>(in_h - 1);

    auto grid_x_scale_t = EigenTensor<T, 3>::From(*grid_x_scale);
    auto grid_y_scale_t = EigenTensor<T, 3>::From(*grid_y_scale);
    grid_grad_x_t = grid_grad_x_t * grid_x_scale_t;
    grid_grad_y_t = grid_grad_y_t * grid_y_scale_t;

    // gather grid_grad [x, y] in 3rd Dim
    T* grid_grad_data = grid_grad->data<T>();
    T* grid_grad_x_data = grid_grad_x.data<T>();
    T* grid_grad_y_data = grid_grad_y.data<T>();
    for (int i = 0; i < n * out_h * out_w; i++) {
      grid_grad_data[2 * i] = grid_grad_x_data[i];
      grid_grad_data[2 * i + 1] = grid_grad_y_data[i];
    }
  }
}

template <typename T>
static void Gather3DBilinearGrad(const CPUContext& ctx,
                                 const DenseTensor& input,
                                 const DenseTensor& output_grad,
                                 DenseTensor* grid_x,
                                 DenseTensor* grid_y,
                                 DenseTensor* grid_z,
                                 DenseTensor* grid_x_scale,
                                 DenseTensor* grid_y_scale,
                                 DenseTensor* grid_z_scale,
                                 DenseTensor* input_grad,
                                 DenseTensor* grid_grad) {
  const int n = grid_x->dims()[0];
  const int out_d = grid_x->dims()[1];
  const int out_h = grid_x->dims()[2];
  const int out_w = grid_x->dims()[3];
  const int c = input.dims()[1];

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
  // gather output grad value to input grad by corner point coords and weight
  Gather3DOutputGradToInputGrad<T>(
      output_grad, input_grad, x_w, y_n, z_t, d_e, d_s, d_b);
  Gather3DOutputGradToInputGrad<T>(
      output_grad, input_grad, x_w, y_s, z_t, d_e, d_n, d_b);
  Gather3DOutputGradToInputGrad<T>(
      output_grad, input_grad, x_e, y_n, z_t, d_w, d_s, d_b);
  Gather3DOutputGradToInputGrad<T>(
      output_grad, input_grad, x_e, y_s, z_t, d_w, d_n, d_b);
  Gather3DOutputGradToInputGrad<T>(
      output_grad, input_grad, x_w, y_n, z_b, d_e, d_s, d_t);
  Gather3DOutputGradToInputGrad<T>(
      output_grad, input_grad, x_w, y_s, z_b, d_e, d_n, d_t);
  Gather3DOutputGradToInputGrad<T>(
      output_grad, input_grad, x_e, y_n, z_b, d_w, d_s, d_t);
  Gather3DOutputGradToInputGrad<T>(
      output_grad, input_grad, x_e, y_s, z_b, d_w, d_n, d_t);

  auto v_twn_t = EigenTensor<T, 5>::From(v_twn);
  auto v_ten_t = EigenTensor<T, 5>::From(v_ten);
  auto v_tws_t = EigenTensor<T, 5>::From(v_tws);
  auto v_tes_t = EigenTensor<T, 5>::From(v_tes);
  auto v_bwn_t = EigenTensor<T, 5>::From(v_bwn);
  auto v_ben_t = EigenTensor<T, 5>::From(v_ben);
  auto v_bws_t = EigenTensor<T, 5>::From(v_bws);
  auto v_bes_t = EigenTensor<T, 5>::From(v_bes);

  auto d_w_t = EigenTensor<T, 4>::From(d_w);
  auto d_e_t = EigenTensor<T, 4>::From(d_e);
  auto d_n_t = EigenTensor<T, 4>::From(d_n);
  auto d_s_t = EigenTensor<T, 4>::From(d_s);
  auto d_t_t = EigenTensor<T, 4>::From(d_t);
  auto d_b_t = EigenTensor<T, 4>::From(d_b);

  auto output_grad_t = EigenTensor<T, 5>::From(output_grad);
  if (grid_grad != nullptr) {
    DenseTensor grid_grad_x, grid_grad_y, grid_grad_z;
    grid_grad_x.Resize({n, out_d, out_h, out_w});
    grid_grad_y.Resize({n, out_d, out_h, out_w});
    grid_grad_z.Resize({n, out_d, out_h, out_w});
    ctx.Alloc<T>(&grid_grad_x);
    ctx.Alloc<T>(&grid_grad_y);
    ctx.Alloc<T>(&grid_grad_z);
    auto grid_grad_x_t =
        EigenTensor<T, 4>::From(grid_grad_x).setConstant(static_cast<T>(0.0));
    auto grid_grad_y_t =
        EigenTensor<T, 4>::From(grid_grad_y).setConstant(static_cast<T>(0.0));
    auto grid_grad_z_t =
        EigenTensor<T, 4>::From(grid_grad_z).setConstant(static_cast<T>(0.0));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < c; j++) {
        for (int m = 0; m < out_d; m++) {
          for (int k = 0; k < out_h; k++) {
            for (int l = 0; l < out_w; l++) {
              grid_grad_x_t(i, m, k, l) +=
                  ((v_ten_t(i, j, m, k, l) - v_twn_t(i, j, m, k, l)) *
                       d_s_t(i, m, k, l) * d_b_t(i, m, k, l) +
                   (v_tes_t(i, j, m, k, l) - v_tws_t(i, j, m, k, l)) *
                       d_n_t(i, m, k, l) * d_b_t(i, m, k, l) +
                   (v_ben_t(i, j, m, k, l) - v_bwn_t(i, j, m, k, l)) *
                       d_s_t(i, m, k, l) * d_t_t(i, m, k, l) +
                   (v_bes_t(i, j, m, k, l) - v_bws_t(i, j, m, k, l)) *
                       d_n_t(i, m, k, l) * d_t_t(i, m, k, l)) *
                  output_grad_t(i, j, m, k, l);
              grid_grad_y_t(i, m, k, l) +=
                  ((v_tws_t(i, j, m, k, l) - v_twn_t(i, j, m, k, l)) *
                       d_e_t(i, m, k, l) * d_b_t(i, m, k, l) +
                   (v_tes_t(i, j, m, k, l) - v_ten_t(i, j, m, k, l)) *
                       d_w_t(i, m, k, l) * d_b_t(i, m, k, l) +
                   (v_bws_t(i, j, m, k, l) - v_bwn_t(i, j, m, k, l)) *
                       d_e_t(i, m, k, l) * d_t_t(i, m, k, l) +
                   (v_bes_t(i, j, m, k, l) - v_ben_t(i, j, m, k, l)) *
                       d_w_t(i, m, k, l) * d_t_t(i, m, k, l)) *
                  output_grad_t(i, j, m, k, l);
              grid_grad_z_t(i, m, k, l) +=
                  ((v_bws_t(i, j, m, k, l) - v_tws_t(i, j, m, k, l)) *
                       d_e_t(i, m, k, l) * d_n_t(i, m, k, l) +
                   (v_bes_t(i, j, m, k, l) - v_tes_t(i, j, m, k, l)) *
                       d_w_t(i, m, k, l) * d_n_t(i, m, k, l) +
                   (v_bwn_t(i, j, m, k, l) - v_twn_t(i, j, m, k, l)) *
                       d_e_t(i, m, k, l) * d_s_t(i, m, k, l) +
                   (v_ben_t(i, j, m, k, l) - v_ten_t(i, j, m, k, l)) *
                       d_w_t(i, m, k, l) * d_s_t(i, m, k, l)) *
                  output_grad_t(i, j, m, k, l);
            }
          }
        }
      }
    }

    auto grid_x_scale_t = EigenTensor<T, 4>::From(*grid_x_scale);
    auto grid_y_scale_t = EigenTensor<T, 4>::From(*grid_y_scale);
    auto grid_z_scale_t = EigenTensor<T, 4>::From(*grid_z_scale);

    grid_grad_x_t = grid_grad_x_t * grid_x_scale_t;
    grid_grad_y_t = grid_grad_y_t * grid_y_scale_t;
    grid_grad_z_t = grid_grad_z_t * grid_z_scale_t;
    // gather grid_grad [x, y, z] in 4rd Dim
    T* grid_grad_data = grid_grad->data<T>();
    T* grid_grad_x_data = grid_grad_x.data<T>();
    T* grid_grad_y_data = grid_grad_y.data<T>();
    T* grid_grad_z_data = grid_grad_z.data<T>();
    for (int i = 0; i < n * out_d * out_h * out_w; i++) {
      grid_grad_data[3 * i] = grid_grad_x_data[i];
      grid_grad_data[3 * i + 1] = grid_grad_y_data[i];
      grid_grad_data[3 * i + 2] = grid_grad_z_data[i];
    }
  }
}

template <typename T>
static void GatherOutputGradToInputGrad(const DenseTensor& output_grad,
                                        DenseTensor* input_grad,
                                        const DenseTensor& x,
                                        const DenseTensor& y) {
  const int n = output_grad.dims()[0];
  const int c = output_grad.dims()[1];
  const int out_h = output_grad.dims()[2];
  const int out_w = output_grad.dims()[3];
  const int in_h = input_grad->dims()[2];
  const int in_w = input_grad->dims()[3];
  auto x_t = EigenTensor<T, 3>::From(x);
  auto y_t = EigenTensor<T, 3>::From(y);
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < out_h; k++) {
      for (int l = 0; l < out_w; l++) {
        if (IsInBound(
                x_t(i, k, l), y_t(i, k, l), (T)(in_w - 1), (T)(in_h - 1))) {
          for (int j = 0; j < c; j++) {
            input_grad_t(i,
                         j,
                         static_cast<int>(round(y_t(i, k, l))),
                         static_cast<int>(round(x_t(i, k, l)))) +=
                output_grad_t(i, j, k, l);
          }
        }
      }
    }
  }
}

template <typename T>
static void Gather3DOutputGradToInputGrad(const DenseTensor& output_grad,
                                          DenseTensor* input_grad,
                                          const DenseTensor& x,
                                          const DenseTensor& y,
                                          const DenseTensor& z) {
  const int n = output_grad.dims()[0];
  const int c = output_grad.dims()[1];
  const int out_d = output_grad.dims()[2];
  const int out_h = output_grad.dims()[3];
  const int out_w = output_grad.dims()[4];
  const int in_d = input_grad->dims()[2];
  const int in_h = input_grad->dims()[3];
  const int in_w = input_grad->dims()[4];
  auto x_t = EigenTensor<T, 4>::From(x);
  auto y_t = EigenTensor<T, 4>::From(y);
  auto z_t = EigenTensor<T, 4>::From(z);
  auto input_grad_t = EigenTensor<T, 5>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 5>::From(output_grad);
  for (int i = 0; i < n; i++) {
    for (int m = 0; m < out_d; m++) {
      for (int k = 0; k < out_h; k++) {
        for (int l = 0; l < out_w; l++) {
          if (IsInBound3D(x_t(i, m, k, l),
                          y_t(i, m, k, l),
                          z_t(i, m, k, l),
                          (T)(in_w - 1),
                          (T)(in_h - 1),
                          (T)(in_d - 1))) {
            for (int j = 0; j < c; j++) {
              input_grad_t(i,
                           j,
                           static_cast<int>(round(z_t(i, m, k, l))),
                           static_cast<int>(round(y_t(i, m, k, l))),
                           static_cast<int>(round(x_t(i, m, k, l)))) +=
                  output_grad_t(i, j, m, k, l);
            }
          }
        }
      }
    }
  }
}

template <typename T, typename Context>
void GridSampleGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& grid,
                          const DenseTensor& out_grid,
                          const std::string& mode,
                          const std::string& padding_mode,
                          bool align_corners,
                          DenseTensor* x_grad,
                          DenseTensor* grid_grad) {
  if (x.dims().size() == 4) {
    const int n = grid.dims()[0];
    const int out_h = grid.dims()[1];
    const int out_w = grid.dims()[2];
    const int c = x.dims()[1];
    const int in_h = x.dims()[2];
    const int in_w = x.dims()[3];

    x_grad->Resize({n, c, in_h, in_w});
    dev_ctx.template Alloc<T>(x_grad);
    phi::funcs::SetConstant<Context, T>()(dev_ctx, x_grad, static_cast<T>(0));

    if (grid_grad != nullptr) {
      grid_grad->Resize({n, out_h, out_w, 2});
      dev_ctx.template Alloc<T>(grid_grad);
      phi::funcs::SetConstant<Context, T>()(
          dev_ctx, grid_grad, static_cast<T>(0));
    }

    DenseTensor grid_x, grid_y;
    DenseTensor grid_x_scale, grid_y_scale;
    CalcGridLocationsWithGrad<T>(dev_ctx,
                                 grid,
                                 in_h,
                                 in_w,
                                 align_corners,
                                 padding_mode,
                                 &grid_x,
                                 &grid_y,
                                 &grid_x_scale,
                                 &grid_y_scale);
    if (mode == "bilinear") {
      GatherBilinearGrad<T>(dev_ctx,
                            x,
                            out_grid,
                            &grid_x,
                            &grid_y,
                            &grid_x_scale,
                            &grid_y_scale,
                            x_grad,
                            grid_grad);
    } else {
      auto grid_x_t = EigenTensor<T, 3>::From(grid_x);
      auto grid_y_t = EigenTensor<T, 3>::From(grid_y);
      grid_x_t = grid_x_t.round();
      grid_y_t = grid_y_t.round();
      GatherOutputGradToInputGrad<T>(out_grid, x_grad, grid_x, grid_y);
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

    x_grad->Resize({n, c, in_d, in_h, in_w});
    dev_ctx.template Alloc<T>(x_grad);
    phi::funcs::SetConstant<Context, T>()(dev_ctx, x_grad, static_cast<T>(0));

    if (grid_grad != nullptr) {
      grid_grad->Resize({n, out_d, out_h, out_w, 3});
      dev_ctx.template Alloc<T>(grid_grad);
      phi::funcs::SetConstant<Context, T>()(
          dev_ctx, grid_grad, static_cast<T>(0));
    }
    DenseTensor grid_x, grid_y, grid_z;
    DenseTensor grid_x_scale, grid_y_scale, grid_z_scale;

    Calc3DGridLocationsWithGrad<T>(dev_ctx,
                                   grid,
                                   in_d,
                                   in_h,
                                   in_w,
                                   align_corners,
                                   padding_mode,
                                   &grid_x,
                                   &grid_y,
                                   &grid_z,
                                   &grid_x_scale,
                                   &grid_y_scale,
                                   &grid_z_scale);
    if (mode == "bilinear") {
      Gather3DBilinearGrad<T>(dev_ctx,
                              x,
                              out_grid,
                              &grid_x,
                              &grid_y,
                              &grid_z,
                              &grid_x_scale,
                              &grid_y_scale,
                              &grid_z_scale,
                              x_grad,
                              grid_grad);
    } else {
      Gather3DOutputGradToInputGrad<T>(
          out_grid, x_grad, grid_x, grid_y, grid_z);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(grid_sample_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::GridSampleGradKernel,
                   float,
                   double) {}
