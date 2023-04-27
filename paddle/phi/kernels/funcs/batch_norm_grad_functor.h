// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {
namespace funcs {

// #ifndef PADDLE_WITH_MKLDNN
template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T, typename Context>
void BatchNormGradFunctor(const Context& ctx,
                          const DenseTensor& x,
                          const DenseTensor& scale,
                          const DenseTensor& bias,
                          const paddle::optional<DenseTensor>& mean,
                          const paddle::optional<DenseTensor>& variance,
                          const DenseTensor& saved_mean,
                          const DenseTensor& saved_variance,
                          const paddle::optional<DenseTensor>& reserve_space,
                          const DenseTensor& y_grad,
                          float momentum,
                          float epsilon,
                          const std::string& data_layout_str,
                          bool is_test,
                          bool use_global_stats,
                          bool trainable_statistics,
                          bool is_inplace,
                          DenseTensor* x_grad,
                          DenseTensor* scale_grad,
                          DenseTensor* bias_grad) {
  const auto* d_y = &y_grad;

  DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

  auto* d_x = x_grad;
  auto* d_scale = scale_grad;
  auto* d_bias = bias_grad;

  use_global_stats = is_test || use_global_stats;

  // batch_norm with inplace as false will take X as grad input, which
  // is same as cuDNN batch_norm backward calculation, batch_norm
  // with inplace as true only take Y as input and X should be calculate
  // by inverse operation of batch_norm on Y

  if (is_inplace) {
    if (d_x) {
      PADDLE_ENFORCE_EQ(d_x,
                        d_y,
                        phi::errors::InvalidArgument(
                            "X@GRAD and Y@GRAD inplaced in non-inplace mode"));
    }
  } else {
    if (d_x) {
      PADDLE_ENFORCE_NE(d_x,
                        d_y,
                        phi::errors::InvalidArgument(
                            "X@GRAD and Y@GRAD inplaced in non-inplace mode"));
    }
  }

  // Get the size for each dimension.
  // NCHW [batch_size, in_channels, in_height, in_width]
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The size of input X's dimensions should be larger than 1."
          "But received: the size of input X's dimensions is [%d]",
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(),
      5,
      phi::errors::InvalidArgument(
          "The size of input X's dimensions should be less than 6."
          "But received: the size of input X's dimensions is [%d]",
          x_dims.size()));
  const int N = x_dims[0];
  const int C = (data_layout == DataLayout::kNCHW ? x_dims[1]
                                                  : x_dims[x_dims.size() - 1]);
  const int sample_size = x.numel() / N / C;

  // input dimension is 2 and the format is NCHW. The input can be regarded as
  // NHWC format
  if (x_dims.size() == 2 && data_layout == DataLayout::kNCHW) {
    data_layout = DataLayout::kNHWC;
  }

  // init output
  if (d_x) {
    ctx.template Alloc<T>(d_x);
  }

  const T* mean_data = nullptr;
  const T* inv_var_data = nullptr;
  DenseTensor inv_var_tensor;
  if (use_global_stats) {
    const auto* running_mean = mean.get_ptr();
    const auto* running_variance = variance.get_ptr();
    mean_data = running_mean->data<T>();
    inv_var_tensor.Resize({C});
    T* running_inv_var_data = ctx.template Alloc<T>(&inv_var_tensor);
    EigenVectorArrayMap<T> inv_var_tmp(running_inv_var_data, C);
    ConstEigenVectorArrayMap<T> var_arr(running_variance->data<T>(), C);

    inv_var_tmp = (var_arr + epsilon).sqrt().inverse();
    inv_var_data = running_inv_var_data;
  } else {
    mean_data = saved_mean.data<T>();
    inv_var_data = saved_variance.data<T>();
  }

  ConstEigenVectorArrayMap<T> scale_arr(scale.data<T>(), C);
  ConstEigenVectorArrayMap<T> bias_arr(bias.data<T>(), C);
  ConstEigenVectorArrayMap<T> mean_arr(mean_data, C);
  ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, C);

  T* d_bias_data = nullptr;
  T* d_scale_data = nullptr;
  if (d_scale && d_bias) {
    d_bias_data = ctx.template Alloc<T>(d_bias);
    d_scale_data = ctx.template Alloc<T>(d_scale);
  }

  // d_bias = np.sum(d_y, axis=0)
  // d_scale = np.sum((X - mean) / inv_std * dy, axis=0)
  // d_x = (1. / N) * scale * inv_var * (N * d_y - np.sum(d_y, axis=0)
  //   - (X - mean) * inv_var * inv_var * np.sum(d_y * (X - mean), axis=0))
  EigenVectorArrayMap<T> d_bias_arr(d_bias_data, C);
  EigenVectorArrayMap<T> d_scale_arr(d_scale_data, C);

  if (d_scale && d_bias) {
    d_bias_arr.setZero();
    d_scale_arr.setZero();
  }

  if (d_x && (N * sample_size) == 1 && !use_global_stats) {
    phi::Copy(ctx, *d_y, ctx.GetPlace(), false, d_x);
    return;
  }

  int scale_coefff = use_global_stats ? 1 : N * sample_size;
  const auto scale_inv_var_nhw = scale_arr * inv_var_arr / scale_coefff;

  DenseTensor dy_sum;
  dy_sum.Resize({C});
  auto dy_sum_data = ctx.template Alloc<T>(&dy_sum);
  EigenVectorArrayMap<T> dy_sum_arr(dy_sum_data, C);

  DenseTensor dy_mul_x_sub_mean_mul_invstd_sum;
  dy_mul_x_sub_mean_mul_invstd_sum.Resize({C});
  auto dy_mul_x_sub_mean_mul_invstd_sum_data =
      ctx.template Alloc<T>(&dy_mul_x_sub_mean_mul_invstd_sum);
  EigenVectorArrayMap<T> dy_mul_x_sub_mean_mul_invstd_sum_arr(
      dy_mul_x_sub_mean_mul_invstd_sum_data, C);

  dy_sum_arr.setZero();
  dy_mul_x_sub_mean_mul_invstd_sum_arr.setZero();

  // inplace calculation
  // Y:  ((x - est_mean) * (inv_var) * scale + bias
  //   formula transform ====>
  //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
  // X: (y - bias) / scale / (inv_var) + est_mean
  //   formula transform ====>
  //    (y - bias) / (scale * inv_var) + est_mean
  switch (data_layout) {
    case DataLayout::kNCHW: {
      if (is_inplace) {
        auto px = x;
        EigenArrayMap<T> x_data(ctx.template Alloc<T>(&px), sample_size, N * C);
        ConstEigenArrayMap<T> y_data(x.data<T>(), sample_size, N * C);
        for (int nc = 0; nc < N * C; ++nc) {
          x_data.col(nc) = (y_data.col(nc) - bias_arr(nc % C)) /
                               scale_inv_var_nhw(nc % C) / scale_coefff +
                           mean_arr(nc % C);
        }
      }
      ConstEigenArrayMap<T> x_arr(x.data<T>(), sample_size, N * C);
      ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), sample_size, N * C);

      for (int nc = 0; nc < N * C; ++nc) {
        int c = nc % C;
        dy_sum_arr(c) += d_y_arr.col(nc).sum();
        dy_mul_x_sub_mean_mul_invstd_sum_arr(c) +=
            ((x_arr.col(nc) - mean_arr(c)) * inv_var_arr(c) * d_y_arr.col(nc))
                .sum();
      }

      if (d_scale && d_bias) {
        d_bias_arr = dy_sum_arr;
        d_scale_arr = dy_mul_x_sub_mean_mul_invstd_sum_arr;
      }

      if (d_x) {
        EigenArrayMap<T> d_x_arr(
            ctx.template Alloc<T>(d_x), sample_size, N * C);
        if (!use_global_stats) {
          for (int nc = 0; nc < N * C; ++nc) {
            int c = nc % C;
            d_x_arr.col(nc) =
                scale_inv_var_nhw(c) *
                (d_y_arr.col(nc) * N * sample_size - dy_sum_arr(c) -
                 (x_arr.col(nc) - mean_arr[c]) *
                     dy_mul_x_sub_mean_mul_invstd_sum_arr(c) * inv_var_arr(c));
          }
        } else {
          for (int nc = 0; nc < N * C; ++nc) {
            int c = nc % C;
            d_x_arr.col(nc) = scale_inv_var_nhw(c) * d_y_arr.col(nc);
          }
        }
      }
      break;
    }
    case DataLayout::kNHWC: {
      if (is_inplace) {
        auto px = x;
        EigenArrayMap<T> x_data(ctx.template Alloc<T>(&px), C, N * sample_size);
        ConstEigenArrayMap<T> y_data(x.data<T>(), C, N * sample_size);
        for (int nhw = 0; nhw < N * sample_size; nhw++) {
          x_data.col(nhw) =
              (y_data.col(nhw) - bias_arr) / scale_inv_var_nhw / scale_coefff +
              mean_arr;
        }
      }
      ConstEigenArrayMap<T> x_arr(x.data<T>(), C, N * sample_size);
      ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), C, N * sample_size);

      for (int nhw = 0; nhw < N * sample_size; ++nhw) {
        dy_sum_arr += d_y_arr.col(nhw);
        dy_mul_x_sub_mean_mul_invstd_sum_arr +=
            (x_arr.col(nhw) - mean_arr) * inv_var_arr * d_y_arr.col(nhw);
      }

      if (d_scale && d_bias) {
        d_bias_arr = dy_sum_arr;
        d_scale_arr = dy_mul_x_sub_mean_mul_invstd_sum_arr;
      }

      if (d_x) {
        EigenArrayMap<T> d_x_arr(
            ctx.template Alloc<T>(d_x), C, N * sample_size);
        if (!use_global_stats) {
          for (int nhw = 0; nhw < N * sample_size; ++nhw) {
            d_x_arr.col(nhw) =
                scale_inv_var_nhw *
                (d_y_arr.col(nhw) * N * sample_size - dy_sum_arr -
                 (x_arr.col(nhw) - mean_arr) *
                     dy_mul_x_sub_mean_mul_invstd_sum_arr * inv_var_arr);
          }
        } else {
          for (int nhw = 0; nhw < N * sample_size; ++nhw) {
            d_x_arr.col(nhw) = scale_inv_var_nhw * d_y_arr.col(nhw);
          }
        }
      }
      break;
    }
    default:
      PADDLE_THROW(phi::errors::InvalidArgument("Unknown storage order: %s",
                                                data_layout_str));
  }
}

}  // namespace funcs
}  // namespace phi
