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

#include "paddle/phi/kernels/batch_norm_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"

namespace phi {

template <typename T>
static int CalculateInvBNY(xpu::Context *ctx,
                           T *x,
                           const T *scale,
                           const T *bias,
                           const T *mean,
                           const T *variance,
                           const int N,
                           const int C,
                           const int M,
                           const T *y) {
  PADDLE_ENFORCE_EQ(x,
                    y,
                    phi::errors::InvalidArgument(
                        "X and Y should be inplaced in inplace mode"));
  std::vector<int> tensor_shape_vec({N, C, M});
  std::vector<int> array_shape_vec({1, C, 1});
  // y - bias
  int r1 =
      xpu::broadcast_sub<T>(ctx, bias, y, x, array_shape_vec, tensor_shape_vec);
  // (y - bias) / scale
  int r2 = xpu::broadcast_div<T>(
      ctx, scale, x, x, array_shape_vec, tensor_shape_vec);
  // (y - bias) / scale / variance
  int r3 = xpu::broadcast_div<T>(
      ctx, variance, x, x, array_shape_vec, tensor_shape_vec);
  // (y - bias) / scale / variance + mean
  int r4 =
      xpu::broadcast_add<T>(ctx, mean, x, x, array_shape_vec, tensor_shape_vec);

  return r1 + r2 + r3 + r4;
}

template <typename T>
static int CalculateInvVar(xpu::Context *ctx,
                           const T *var,
                           const T epsilon,
                           const int C,
                           T *epsilon_data,
                           T *inv_var) {
  int r1 = constant(ctx, epsilon_data, 1, epsilon);
  std::vector<int> tensor_shape_vec({C});
  std::vector<int> array_shape_vec({1});
  int r2 = xpu::broadcast_add<T>(
      ctx, epsilon_data, var, inv_var, array_shape_vec, tensor_shape_vec);
  int r3 = xpu::rsqrt<T>(ctx, inv_var, inv_var, C);
  return r1 + r2 + r3;
}

template <typename T, typename Context>
void BatchNormGradKernel(const Context &dev_ctx,
                         const DenseTensor &x,
                         const DenseTensor &scale,
                         const DenseTensor &bias,
                         const paddle::optional<DenseTensor> &mean,
                         const paddle::optional<DenseTensor> &variance,
                         const DenseTensor &saved_mean,
                         const DenseTensor &saved_variance,
                         const paddle::optional<DenseTensor> &reserve_space,
                         const DenseTensor &y_grad,
                         float momentum,
                         float epsilon,
                         const std::string &data_layout,
                         bool is_test,
                         bool use_global_stats,
                         bool trainable_statistics,
                         bool fuse_with_relu,
                         DenseTensor *x_grad,
                         DenseTensor *scale_grad,
                         DenseTensor *bias_grad) {
  const auto *d_y = &y_grad;
  PADDLE_ENFORCE_EQ(data_layout == "NCHW" || data_layout == "NHWC",
                    true,
                    phi::errors::InvalidArgument(
                        "The 'data_layout' attribute must be NCHW or NHWC. "
                        "But recevived 'data_layout' is [%s].",
                        data_layout));

  const auto data_layout_val =
      paddle::framework::StringToDataLayout(data_layout);

  use_global_stats = is_test || use_global_stats;

  // batch_norm with inplace as false will take X as grad input, which
  // is same as cuDNN batch_norm backward calculation, batch_norm
  // with inplace as true only take Y as input and X should be calculate
  // by inverse operation of batch_norm on Y
  bool is_inplace = false;
  if (x_grad) {
    PADDLE_ENFORCE_NE(x_grad,
                      d_y,
                      phi::errors::InvalidArgument(
                          "X@GRAD and Y@GRAD inplaced in non-inplace mode"));
  }

  const auto &x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      phi::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  int N = -1, C = -1, H = -1, W = -1, D = -1;
  funcs::ExtractNCWHD(x_dims, data_layout_val, &N, &C, &H, &W, &D);
  N = (N == 0) ? 1 : N;
  C = (C == 0) ? 1 : C;
  H = (H == 0) ? 1 : H;
  W = (W == 0) ? 1 : W;

  const auto *x_data = x.data<T>();
  const auto *d_y_data = y_grad.data<T>();
  const auto *scale_data = scale.data<float>();

  // init output
  T *x_grad_data = nullptr;
  T *bias_grad_data = nullptr;
  T *scale_grad_data = nullptr;
  if (x_grad) {
    x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  }
  if (scale_grad && bias_grad) {
    scale_grad_data = dev_ctx.template Alloc<T>(scale_grad);
    bias_grad_data = dev_ctx.template Alloc<T>(bias_grad);
  }

  PADDLE_ENFORCE_EQ(
      scale.dims().size(),
      1UL,
      phi::errors::InvalidArgument(
          "The size of scale's dimensions must equal to 1. But received: "
          "the size of scale's dimensions is [%d], the dimensions of scale "
          "is [%s].",
          scale.dims().size(),
          scale.dims()));
  PADDLE_ENFORCE_EQ(
      scale.dims()[0],
      C,
      phi::errors::InvalidArgument(
          "The first dimension of scale must equal to Channels[%d]. But "
          "received: the first dimension of scale is [%d]",
          C,
          scale.dims()[0]));

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  const auto *global_mean = mean.get_ptr();
  const auto *global_var = variance.get_ptr();

  // TODO(guozibin): hadle the situation case of N * H * W = 1
  if (is_inplace) {
    float *global_inv_std_data = nullptr;
    if (use_global_stats) {
      global_inv_std_data =
          RAII_GUARD.alloc_l3_or_gm<float>(global_var->numel());
      float *epsilon_data = RAII_GUARD.alloc_l3_or_gm<float>(1);
      int r1 = CalculateInvVar(dev_ctx.x_context(),
                               global_var->data<float>(),
                               epsilon,
                               C,
                               epsilon_data,
                               global_inv_std_data);
      PADDLE_ENFORCE_XDNN_SUCCESS(r1,
                                  "batch_norm_grad CalculateInvVar function");
    }

    // Here is a trick, x is a const input,
    // but trans to a non-const var, is it risky?
    auto px = x;
    auto *inv_std_data =
        use_global_stats ? global_inv_std_data : saved_variance.data<float>();
    auto *mean_data = use_global_stats ? global_mean->data<float>()
                                       : saved_mean.data<float>();
    int r2 = CalculateInvBNY(dev_ctx.x_context(),
                             px.data<T>(),
                             scale.data<float>(),
                             bias.data<float>(),
                             mean_data,
                             inv_std_data,
                             N,
                             C,
                             H * W,
                             x.data<T>());
    PADDLE_ENFORCE_XDNN_SUCCESS(r2, "batch_norm_grad CalculateInvBNY function");
  }

  int r3;
  bool is_nchw = data_layout == "NCHW";
  if (use_global_stats) {
    r3 = xpu::batch_norm_grad<T>(dev_ctx.x_context(),
                                 x_data,
                                 d_y_data,
                                 x_grad_data,
                                 N,
                                 C,
                                 H,
                                 W,
                                 scale_data,
                                 nullptr,
                                 nullptr,
                                 scale_grad_data,
                                 bias_grad_data,
                                 is_nchw,
                                 global_mean->data<float>(),
                                 global_var->data<float>(),
                                 epsilon);
  } else {
    if (!x_grad) {
      x_grad_data = RAII_GUARD.alloc_l3_or_gm<T>(x.numel());
    }
    if (!scale_grad) {
      scale_grad_data = RAII_GUARD.alloc_l3_or_gm<float>(C);
    }
    if (!bias_grad_data) {
      bias_grad_data = RAII_GUARD.alloc_l3_or_gm<float>(C);
    }
    r3 = xpu::batch_norm_grad<T>(dev_ctx.x_context(),
                                 x_data,
                                 d_y_data,
                                 x_grad_data,
                                 N,
                                 C,
                                 H,
                                 W,
                                 scale_data,
                                 saved_mean.data<float>(),
                                 saved_variance.data<float>(),
                                 scale_grad_data,
                                 bias_grad_data,
                                 is_nchw);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r3, "batch_norm_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    batch_norm_grad, XPU, ALL_LAYOUT, phi::BatchNormGradKernel, float) {}
