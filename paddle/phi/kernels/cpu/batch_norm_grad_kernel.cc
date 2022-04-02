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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/batch_norm_utils.h"

namespace phi {

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
void BatchNormGradRawKernel(const Context& ctx,

                            const DenseTensor& x,
                            const DenseTensor& scale,
                            const DenseTensor& bias,
                            paddle::optional<const DenseTensor&> mean,
                            paddle::optional<const DenseTensor&> variance,
                            const DenseTensor& saved_mean,
                            const DenseTensor& saved_variance,
                            paddle::optional<const DenseTensor&> reserve_space,
                            const DenseTensor& y_grad,
                            float momentum,
                            float epsilon,
                            const std::string& data_layout_str,
                            bool is_test,
                            bool use_global_stats,
                            bool trainable_statistics,
                            bool fuse_with_relu,
                            bool is_inplace,
                            DenseTensor* x_grad,
                            DenseTensor* scale_grad,
                            DenseTensor* bias_grad) {
  const auto* d_y = &y_grad;

  DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);

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
    paddle::framework::TensorCopy(*d_y, ctx.GetPlace(), d_x);
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

template <typename T, typename Context>
void BatchNormGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& scale,
                         const DenseTensor& bias,
                         paddle::optional<const DenseTensor&> mean,
                         paddle::optional<const DenseTensor&> variance,
                         const DenseTensor& saved_mean,
                         const DenseTensor& saved_variance,
                         paddle::optional<const DenseTensor&> reserve_space,
                         const DenseTensor& y_grad,
                         float momentum,
                         float epsilon,
                         const std::string& data_layout,
                         bool is_test,
                         bool use_global_stats,
                         bool trainable_statistics,
                         bool fuse_with_relu,
                         DenseTensor* x_grad,
                         DenseTensor* scale_grad,
                         DenseTensor* bias_grad) {
  BatchNormGradRawKernel<T, Context>(dev_ctx,
                                     x,
                                     scale,
                                     bias,
                                     mean,
                                     variance,
                                     saved_mean,
                                     saved_variance,
                                     reserve_space,
                                     y_grad,
                                     momentum,
                                     epsilon,
                                     data_layout,
                                     is_test,
                                     use_global_stats,
                                     trainable_statistics,
                                     fuse_with_relu,
                                     false,
                                     x_grad,
                                     scale_grad,
                                     bias_grad);
}

template <typename T, typename Context>
void BatchNormDoubleGradKernel(const Context& ctx,
                               const DenseTensor& x_grad_grad,
                               const DenseTensor& scale_grad_grad,
                               const DenseTensor& bias_grad_grad,
                               const DenseTensor& y_grad,
                               const DenseTensor& x,
                               const DenseTensor& scale,
                               const DenseTensor& saved_mean,
                               const DenseTensor& saved_variance,
                               paddle::optional<const DenseTensor&> mean,
                               paddle::optional<const DenseTensor&> variance,
                               float momentum,
                               float epsilon,
                               const std::string& data_layout_str,
                               bool is_test,
                               bool use_global_stats,
                               bool trainable_statistics,
                               bool fuse_with_relu,
                               DenseTensor* x_grad,
                               DenseTensor* scale_grad,
                               DenseTensor* y_grad_grad) {
  const auto* X = &x;
  const auto* Scale = &scale;
  const auto* dY = &y_grad;
  const auto* Saved_mean = &saved_mean;
  const auto* Saved_variance = &saved_variance;

  PADDLE_ENFORCE_EQ(is_test,
                    false,
                    phi::errors::InvalidArgument(
                        "`is_test = True` CANNOT be used in train program. If "
                        "you want to use global status in pre_train model, "
                        "please set `use_global_stats = True`"));

  const auto data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);

  const auto* ddX = &x_grad_grad;
  const auto* ddScale = &scale_grad_grad;
  const auto* ddBias = &bias_grad_grad;

  auto* dX = x_grad;
  auto* dScale = scale_grad;
  auto* ddY = y_grad_grad;
  ctx.template Alloc<T>(dX);
  ctx.template Alloc<T>(ddY);

  const auto& x_dims = X->dims();
  const int C = (data_layout == DataLayout::kNCHW ? x_dims[1]
                                                  : x_dims[x_dims.size() - 1]);
  const int sample_size = X->numel() / C;
  phi::funcs::SetConstant<Context, T> set_constant;

  const T* mean_data = Saved_mean->data<T>();
  const T* inv_var_data = Saved_variance->data<T>();

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
  }

  // transpose NCHW -> NHWC for easy calculate
  DenseTensor transformed_x(X->type());
  DenseTensor transformed_dy(dY->type());
  DenseTensor transformed_ddx(ddX->type());

  DenseTensor transformed_dx(dX->type());
  DenseTensor transformed_ddy(ddY->type());
  if (data_layout == DataLayout::kNCHW && x_dims.size() > 2) {
    VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
    // Input Tensor
    ResizeToChannelLast<Context, T>(ctx, X, &transformed_x);
    TransToChannelLast<Context, T>(ctx, X, &transformed_x);
    ResizeToChannelLast<Context, T>(ctx, dY, &transformed_dy);
    TransToChannelLast<Context, T>(ctx, dY, &transformed_dy);
    ResizeToChannelLast<Context, T>(ctx, ddX, &transformed_ddx);
    TransToChannelLast<Context, T>(ctx, ddX, &transformed_ddx);
    // Output Tensor
    ResizeToChannelLast<Context, T>(ctx, dX, &transformed_dx);
    ResizeToChannelLast<Context, T>(ctx, ddY, &transformed_ddy);
  } else {
    transformed_x.ShareDataWith(*X);
    transformed_dy.ShareDataWith(*dY);
    transformed_ddx.ShareDataWith(*ddX);

    transformed_dx.ShareDataWith(*dX);
    transformed_ddy.ShareDataWith(*ddY);
  }

  ConstEigenArrayMap<T> x_arr(transformed_x.data<T>(), C, sample_size);
  ConstEigenVectorArrayMap<T> mean_arr(mean_data, C);
  ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, C);

  Tensor mean_tile;
  mean_tile.Resize({C, sample_size});
  EigenArrayMap<T> mean_tile_data(
      ctx.template Alloc<T>(&mean_tile), C, sample_size);

  DenseTensor inv_var_tile;
  inv_var_tile.Resize({C, sample_size});
  EigenArrayMap<T> inv_var_tile_data(
      ctx.template Alloc<T>(&inv_var_tile), C, sample_size);

  mean_tile_data = mean_arr.replicate(1, sample_size);
  inv_var_tile_data = inv_var_arr.replicate(1, sample_size);

  DenseTensor Scale_data;
  if (!Scale) {
    Scale_data.Resize({C});
    ctx.template Alloc<T>(&Scale_data);
    set_constant(ctx, &Scale_data, static_cast<T>(1));
  }
  ConstEigenVectorArrayMap<T> scale_arr(
      Scale ? Scale->data<T>() : Scale_data.data<T>(), C);

  Tensor scale_tile;
  scale_tile.Resize({C, sample_size});
  EigenArrayMap<T> scale_tile_data(
      ctx.template Alloc<T>(&scale_tile), C, sample_size);
  scale_tile_data = scale_arr.replicate(1, sample_size);

  ConstEigenArrayMap<T> dy_arr(transformed_dy.data<T>(), C, sample_size);
  ConstEigenArrayMap<T> ddx_arr(transformed_ddx.data<T>(), C, sample_size);

  DenseTensor x_sub_mean_mul_invstd;
  x_sub_mean_mul_invstd.Resize({C, sample_size});

  EigenArrayMap<T> x_sub_mean_mul_invstd_arr(
      ctx.template Alloc<T>(&x_sub_mean_mul_invstd), C, sample_size);
  x_sub_mean_mul_invstd_arr = (x_arr - mean_tile_data) * inv_var_tile_data;

  if (dX) {
    ctx.template Alloc<T>(dX);
    EigenArrayMap<T> dx_arr(
        ctx.template Alloc<T>(&transformed_dx), C, sample_size);
    dx_arr.setZero();
    if (use_global_stats) {
      // math: dx = (ddscale * dy) * inv_var
      if (ddScale) {
        ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
        Tensor ddscale_tile;
        ddscale_tile.Resize({C, sample_size});
        EigenArrayMap<T> ddscale_tile_data(
            ctx.template Alloc<T>(&ddscale_tile), C, sample_size);
        ddscale_tile_data = ddscale_arr.replicate(1, sample_size);

        dx_arr = dy_arr * ddscale_tile_data * inv_var_tile_data;
      }
    } else {
      // math: dx = scale * ((x - mean) * inv_var / NxHxW * (np.mean(ddx,
      // axis=(n,h,w)) *
      //          np.sum(dy, axis=(n,h,w)) -
      //          np.sum(dy * ddx, axis=(n,h,w)) + 3 * np.mean(dy * (x -
      //          mean),
      //          axis=(n,h,w)) * inv_var.pow(2) *
      //          np.sum(ddx * (x - mean), axis=(n,h,w))) + inv_var.pow(3) /
      //          NxHxW *
      //          np.sum(ddx * (x - mean)) *
      //          (np.mean(dy, axis=(n,h,w)) - dy) + inv_var.pow(3) / NxHxW *
      //          np.sum(dy,
      //          axis=(n,h,w)) * (x - mean) *
      //          (np.mean(ddx, axis=(n,h,w)) - ddx)) + ddr * (dy * inv_var -
      //          inv_var
      //          *
      //          np.mean(dy, axis=(n,h,w)) -
      //          inv_var.pow(3) * (x - mean) * np.mean(dy * (x - mean),
      //          axis=(n,h,w)))

      if (ddX) {
        dx_arr +=
            (x_sub_mean_mul_invstd_arr * inv_var_tile_data * inv_var_tile_data /
             sample_size)
                .colwise() *
            (ddx_arr.rowwise().sum() * dy_arr.rowwise().sum() / sample_size -
             (dy_arr * ddx_arr).rowwise().sum() +
             3. * (dy_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() *
                 (ddx_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() /
                 sample_size);

        dx_arr += (inv_var_tile_data * inv_var_tile_data).colwise() *
                  (ddx_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() /
                  sample_size * (dy_arr.rowwise().sum() / sample_size - dy_arr);

        dx_arr += (inv_var_tile_data * inv_var_tile_data).colwise() *
                  (dy_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() /
                  sample_size *
                  (ddx_arr.rowwise().sum() / sample_size - ddx_arr);

        dx_arr = scale_tile_data * dx_arr;
      }
      if (ddScale) {
        ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
        Tensor ddscale_tile;
        ddscale_tile.Resize({C, sample_size});
        EigenArrayMap<T> ddscale_tile_data(
            ctx.template Alloc<T>(&ddscale_tile), C, sample_size);
        ddscale_tile_data = ddscale_arr.replicate(1, sample_size);

        dx_arr +=
            (dy_arr * inv_var_tile_data -
             (dy_arr.rowwise().sum().replicate(1, sample_size) / sample_size) *
                 inv_var_tile_data -
             x_sub_mean_mul_invstd_arr * inv_var_tile_data *
                 (dy_arr * x_sub_mean_mul_invstd_arr)
                     .rowwise()
                     .sum()
                     .replicate(1, sample_size) /
                 sample_size) *
            ddscale_tile_data;
      }
    }
    if (data_layout == DataLayout::kNCHW) {
      VLOG(3) << "Transform batchnorm output from NHWC to NCHW";
      TransToChannelFirst<Context, T>(ctx, &transformed_dx, dX);
    }
  }
  if (dScale) {
    EigenVectorArrayMap<T> dscale_arr(ctx.template Alloc<T>(dScale), C);
    dscale_arr.setZero();
    if (use_global_stats) {
      // math: dscale = np.sum(ddx * dy, axis=(n,h,w)) * inv_var
      if (ddX) {
        dscale_arr = (ddx_arr * dy_arr * inv_var_tile_data).rowwise().sum();
      }
    } else {
      // math: dscale = inv_var * (dy - np.mean(dy, axis=(n,h,w) - (x-mean) *
      //            inv_var.pow(2) * np.mean(dy * (x-mean), axis=(n,h,w)))) *
      //            ddx
      if (ddX) {
        Tensor first_grad;
        first_grad.Resize({C, sample_size});
        EigenArrayMap<T> first_grad_arr(
            ctx.template Alloc<T>(&first_grad), C, sample_size);
        first_grad_arr.setZero();

        first_grad_arr +=
            inv_var_tile_data *
            (dy_arr -
             dy_arr.rowwise().sum().replicate(1, sample_size) / sample_size -
             x_sub_mean_mul_invstd_arr *
                 (dy_arr * x_sub_mean_mul_invstd_arr)
                     .rowwise()
                     .sum()
                     .replicate(1, sample_size) /
                 sample_size);
        dscale_arr = (first_grad_arr * ddx_arr).rowwise().sum();
      }
    }
  }

  if (ddY) {
    ctx.template Alloc<T>(ddY);
    EigenArrayMap<T> ddy_arr(
        ctx.template Alloc<T>(&transformed_ddy), C, sample_size);
    ddy_arr.setZero();
    if (use_global_stats) {
      // math: ddy = r * ddx * inv_var + ddbias +
      //           ddscale * (x - mean) * inv_var
      if (ddX) {
        ddy_arr = scale_tile_data * ddx_arr * inv_var_tile_data;
      }
    } else {
      // math: ddy = (x - mean) * inv_var * ddscale + ddbias +
      //           scale * inv_var * (ddx - (x - mean) * inv_var.pow(2) *
      //           np.mean(ddx * (x - mean), axis=(n,h,w)))
      if (ddX) {
        ddy_arr +=
            scale_tile_data * inv_var_tile_data *
            (ddx_arr -
             ddx_arr.rowwise().sum().replicate(1, sample_size) / sample_size -
             x_sub_mean_mul_invstd_arr *
                 (ddx_arr * x_sub_mean_mul_invstd_arr)
                     .rowwise()
                     .sum()
                     .replicate(1, sample_size) /
                 sample_size);
      }
    }
    if (ddScale) {
      ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
      Tensor ddscale_tile;
      ddscale_tile.Resize({C, sample_size});
      EigenArrayMap<T> ddscale_tile_data(
          ctx.template Alloc<T>(&ddscale_tile), C, sample_size);
      ddscale_tile_data = ddscale_arr.replicate(1, sample_size);

      ddy_arr += x_sub_mean_mul_invstd_arr * ddscale_tile_data;
    }

    if (ddBias) {
      ConstEigenVectorArrayMap<T> ddbias_arr(ddBias->data<T>(), C);
      Tensor ddbias_tile;
      ddbias_tile.Resize({C, sample_size});
      EigenArrayMap<T> ddbias_tile_data(
          ctx.template Alloc<T>(&ddbias_tile), C, sample_size);
      ddbias_tile_data = ddbias_arr.replicate(1, sample_size);

      ddy_arr += ddbias_tile_data;
    }

    if (data_layout == DataLayout::kNCHW) {
      VLOG(3) << "Transform batchnorm output from NHWC to NCHW";
      TransToChannelFirst<Context, T>(ctx, &transformed_ddy, ddY);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    batch_norm_grad, CPU, ALL_LAYOUT, phi::BatchNormGradKernel, float, double) {
}

PD_REGISTER_KERNEL(batch_norm_grad_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::BatchNormGradRawKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(batch_norm_grad_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::BatchNormDoubleGradKernel,
                   float,
                   double) {}
