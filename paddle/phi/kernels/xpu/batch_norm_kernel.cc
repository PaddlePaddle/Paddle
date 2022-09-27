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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"

namespace phi {

template <typename T, typename Context>
void BatchNormKernel(const Context& dev_ctx,
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
  const auto data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);
  PADDLE_ENFORCE_EQ(data_layout_str == "NCHW" || data_layout_str == "NHWC",
                    true,
                    phi::errors::InvalidArgument(
                        "The 'data_layout' attribute must be NCHW or NHWC. "
                        "But recevived 'data_layout' is [%s].",
                        data_layout_str));

  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      phi::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  int N = -1, C = -1, H = -1, W = -1, D = -1;
  funcs::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);
  N = (N == 0) ? 1 : N;
  C = (C == 0) ? 1 : C;
  H = (H == 0) ? 1 : H;
  W = (W == 0) ? 1 : W;

  const auto* x_data = x.data<T>();
  const auto* scale_data = scale.data<float>();
  const auto* bias_data = bias.data<float>();

  // alloc memory
  auto* y_data = dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<float>(mean_out);
  dev_ctx.template Alloc<float>(variance_out);
  dev_ctx.template Alloc<float>(saved_mean);
  dev_ctx.template Alloc<float>(saved_variance);

  bool is_nchw = data_layout_str == "NCHW";

  if (!global_stats) {
    auto* mean_out_data = mean_out->data<float>();
    auto* variance_out_data = variance_out->data<float>();
    auto* saved_mean_data = saved_mean->data<float>();
    auto* saved_variance_data = saved_variance->data<float>();

    int r = xpu::batch_norm<T>(dev_ctx.x_context(),
                               x_data,
                               y_data,
                               N,
                               C,
                               H,
                               W,
                               epsilon,
                               momentum,
                               scale_data,
                               bias_data,
                               saved_mean_data,
                               saved_variance_data,
                               mean_out_data,
                               variance_out_data,
                               is_nchw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm");
  } else {
    const auto* mean_data = mean.data<float>();
    const auto* variance_data = variance.data<float>();
    int r = xpu::batch_norm_infer(dev_ctx.x_context(),
                                  x_data,
                                  y_data,
                                  N,
                                  C,
                                  H,
                                  W,
                                  epsilon,
                                  scale_data,
                                  bias_data,
                                  mean_data,
                                  variance_data,
                                  is_nchw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_infer");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(batch_norm, XPU, ALL_LAYOUT, phi::BatchNormKernel, float) {}
