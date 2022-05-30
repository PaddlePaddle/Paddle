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

#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

#include "paddle/fluid/framework/tensor_util.h"

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
void BatchNormKernel(const Context& ctx,
                     const DenseTensor& x,
                     const DenseTensor& scale,
                     const DenseTensor& bias,
                     const DenseTensor& mean,
                     const DenseTensor& variance,
                     float momentum,
                     float epsilon,
                     const std::string& data_layout_str,
                     bool is_test,
                     bool use_global_stats,
                     bool trainable_statistics,
                     bool fuse_with_relu,
                     DenseTensor* y,
                     DenseTensor* mean_out,
                     DenseTensor* variance_out,
                     DenseTensor* saved_mean,
                     DenseTensor* saved_variance,
                     DenseTensor* reserve_space) {
  bool test_mode = is_test && (!trainable_statistics);

  bool global_stats = test_mode || use_global_stats;

  auto data_layout = paddle::framework::StringToDataLayout(data_layout_str);

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
          "But received: the size of input X's dimensionss is [%d]",
          x_dims.size()));
  const int N = x_dims[0];
  const int C = (data_layout == DataLayout::kNCHW ? x_dims[1]
                                                  : x_dims[x_dims.size() - 1]);
  const int sample_size = x.numel() / N / C;

  // alloc memory
  ctx.template Alloc<T>(y);
  ctx.template Alloc<T>(mean_out);
  ctx.template Alloc<T>(variance_out);
  ctx.template Alloc<T>(saved_mean);
  ctx.template Alloc<T>(saved_variance);

  // input dimension is 2 and the format is NCHW. The input can be regarded
  // as NHWC format
  if (x_dims.size() == 2 && data_layout == DataLayout::kNCHW) {
    data_layout = DataLayout::kNHWC;
  }

  if (!global_stats) {
    // saved_xx is use just in this batch of data
    EigenVectorArrayMap<T> saved_mean_e(ctx.template Alloc<T>(saved_mean), C);
    EigenVectorArrayMap<T> saved_variance_e(
        ctx.template Alloc<T>(saved_variance), C);
    saved_mean_e.setZero();
    saved_variance_e.setZero();

    EigenVectorArrayMap<T> running_mean_arr(ctx.template Alloc<T>(mean_out), C);
    EigenVectorArrayMap<T> running_var_arr(ctx.template Alloc<T>(variance_out),
                                           C);

    if ((N * sample_size) == 1) {
      // Only 1 element in normalization dimension,
      // we skip the batch norm calculation, let y = x.
      paddle::framework::TensorCopy(x, ctx.GetPlace(), y);
      return;
    }

    switch (data_layout) {
      case DataLayout::kNCHW: {
        ConstEigenArrayMap<T> x_arr(x.data<T>(), sample_size, N * C);
        for (int nc = 0; nc < N * C; ++nc) {
          saved_mean_e(nc % C) += x_arr.col(nc).sum();
        }
        saved_mean_e /= N * sample_size;
        for (int nc = 0; nc < N * C; ++nc) {
          saved_variance_e(nc % C) +=
              (x_arr.col(nc) - saved_mean_e(nc % C)).matrix().squaredNorm();
        }
        saved_variance_e /= N * sample_size;
        break;
      }
      case DataLayout::kNHWC: {
        ConstEigenArrayMap<T> x_arr(x.data<T>(), C, N * sample_size);
        for (int i = 0; i < N * sample_size; ++i) {
          saved_mean_e += x_arr.col(i);
        }
        saved_mean_e /= N * sample_size;
        for (int i = 0; i < N * sample_size; ++i) {
          saved_variance_e +=
              (x_arr.col(i) - saved_mean_e) * (x_arr.col(i) - saved_mean_e);
        }
        saved_variance_e /= N * sample_size;
        break;
      }
      default:
        PADDLE_THROW(phi::errors::InvalidArgument("Unknown storage order: %s",
                                                  data_layout_str));
    }

    // if MomentumTensor is set, use MomentumTensor value, momentum
    // is only used in this training branch

    running_mean_arr =
        running_mean_arr * momentum + saved_mean_e * (1. - momentum);
    running_var_arr =
        running_var_arr * momentum + saved_variance_e * (1. - momentum);
  }

  // use SavedMean and SavedVariance to do normalize
  Eigen::Array<T, Eigen::Dynamic, 1> inv_std(C);
  if (global_stats) {
    ConstEigenVectorArrayMap<T> var_arr(variance.data<T>(), C);
    inv_std = (var_arr + epsilon).sqrt().inverse();
  } else {
    EigenVectorArrayMap<T> saved_inv_std(saved_variance->data<T>(), C);
    // inverse SavedVariance first, gradient will use it too.
    saved_inv_std = (saved_inv_std + epsilon).inverse().sqrt();
    inv_std = saved_inv_std;
  }
  ConstEigenVectorArrayMap<T> mean_arr(
      global_stats ? mean.data<T>() : saved_mean->data<T>(), C);

  //   ((x - est_mean) * (inv_var) * scale + bias
  //   formula transform ====>
  //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
  ConstEigenVectorArrayMap<T> scale_arr(scale.data<T>(), C);
  ConstEigenVectorArrayMap<T> bias_arr(bias.data<T>(), C);
  Eigen::Array<T, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
  Eigen::Array<T, Eigen::Dynamic, 1> new_bias =
      bias_arr - mean_arr * inv_std * scale_arr;

  switch (data_layout) {
    case DataLayout::kNCHW: {
      EigenArrayMap<T> y_arr(ctx.template Alloc<T>(y), sample_size, N * C);
      ConstEigenArrayMap<T> x_arr(x.data<T>(), sample_size, N * C);
      for (int nc = 0; nc < N * C; ++nc) {
        y_arr.col(nc) = x_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
      }
      break;
    }
    case DataLayout::kNHWC: {
      EigenArrayMap<T>(ctx.template Alloc<T>(y), C, N * sample_size) =
          (ConstEigenArrayMap<T>(x.data<T>(), C, N * sample_size).colwise() *
           new_scale)
              .colwise() +
          new_bias;
      break;
    }
    default:
      PADDLE_THROW(phi::errors::InvalidArgument("Unknown storage order: %d",
                                                data_layout));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    batch_norm, CPU, ALL_LAYOUT, phi::BatchNormKernel, float, double) {}
