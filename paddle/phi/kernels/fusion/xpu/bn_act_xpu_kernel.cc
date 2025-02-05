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

#include "glog/logging.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void BNActXPUKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& mean,
                    const DenseTensor& variance,
                    const DenseTensor& scale,
                    const DenseTensor& bias,
                    float momentum,
                    float epsilon,
                    const std::string& data_layout_str,
                    int act_type,
                    DenseTensor* y) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto data_layout = common::StringToDataLayout(data_layout_str);
  PADDLE_ENFORCE_EQ(data_layout_str == "NCHW" || data_layout_str == "NHWC",
                    true,
                    common::errors::InvalidArgument(
                        "The 'data_layout' attribute must be NCHW or NHWC. "
                        "But received 'data_layout' is [%s].",
                        data_layout_str));

  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      common::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));
  int N = -1, C = -1, H = -1, W = -1, D = -1;
  funcs::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);
  N = (N == 0) ? 1 : N;
  C = (C == 0) ? 1 : C;
  H = (H == 0) ? 1 : H;
  W = (W == 0) ? 1 : W;
  D = (D == 0) ? 1 : D;

  W = W * D;
  const auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  const auto* scale_data = scale.data<float>();
  const auto* bias_data = bias.data<float>();

  // alloc memory
  auto* y_data = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(y));
  PADDLE_ENFORCE_LE(
      x_dims.size(),
      5,
      common::errors::InvalidArgument(
          "The size of input X's dimensions should be less than 6."
          "But received: the size of input X's dimensions is [%d]",
          x_dims.size()));

  bool is_nchw = data_layout_str == "NCHW";
  const auto* mean_data = mean.data<float>();
  const auto* variance_data = variance.data<float>();
#ifndef PADDLE_WITH_XPU_PLUGIN
  LOG(WARNING) << "Add -DWITH_XPU_PLUGIN=ON to build "
                  "xpu::plugin::bn_act_fusion_infer(), which will lead high "
                  "performance.";
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  XPUType* temp_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(x.numel());
  int r = xpu::batch_norm_infer<XPUType>(dev_ctx.x_context(),
                                         x_data,
                                         temp_data,
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
  r = xpu::relu(
      dev_ctx.x_context(), temp_data, y_data, x.numel(), nullptr, nullptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu");
#else
  int r = xpu::plugin::bn_act_fusion_infer<XPUType>(dev_ctx.x_context(),
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
                                                    is_nchw,
                                                    act_type);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "bn_act_fusion_infer");
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(bn_act_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::BNActXPUKernel,
                   float,
                   phi::dtype::float16) {}
