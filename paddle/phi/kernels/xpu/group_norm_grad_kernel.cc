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

#include "paddle/phi/kernels/group_norm_grad_kernel.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <string>

#include "paddle/common/layout.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void GroupNormGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& scale,
                         const paddle::optional<DenseTensor>& bias,
                         const DenseTensor& y,
                         const DenseTensor& mean,
                         const DenseTensor& var,
                         const DenseTensor& d_y,
                         float epsilon,
                         int groups,
                         const std::string& data_layout_str,
                         DenseTensor* d_x,
                         DenseTensor* d_scale,
                         DenseTensor* d_bias) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(d_x);
    if (d_scale) {
      // If batch dim is 0, we should set d_scale to zero, or else NAN
      if (x.dims().size() > 0 && x.dims()[0] == 0) {
        phi::Full<T, Context>(dev_ctx,
                              phi::IntArray(common::vectorize(d_scale->dims())),
                              0,
                              d_scale);

      } else {
        phi::Full<T, Context>(dev_ctx,
                              phi::IntArray(common::vectorize(d_scale->dims())),
                              NAN,
                              d_scale);
      }
    }
    if (d_bias) {
      phi::Full<T, Context>(
          dev_ctx, phi::IntArray(common::vectorize(d_bias->dims())), 0, d_bias);
    }
    return;
  }
  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int ret = 0;
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();
  const auto x_dims = common::vectorize<int64_t>(x.dims());
  const int64_t N = x_dims[0];
  const bool channel_first =
      data_layout == DataLayout::kNCHW || data_layout == DataLayout::kNCDHW;
  const int64_t C = (channel_first ? x_dims[1] : x_dims[x_dims.size() - 1]);
  const int64_t L =
      (channel_first ? std::accumulate(x_dims.begin() + 2,
                                       x_dims.end(),
                                       1,
                                       std::multiplies<int64_t>())
                     : std::accumulate(x_dims.begin() + 1,
                                       x_dims.end() - 1,
                                       1,
                                       std::multiplies<int64_t>()));

  dev_ctx.template Alloc<T>(d_x);
  phi::funcs::SetConstant<XPUContext, T> set_zero;

  auto* x_data = x.data<T>();
  auto* y_data = y.data<T>();
  auto* d_x_data = d_x->data<T>();
  auto* d_y_data = d_y.data<T>();
  T* d_scale_data = nullptr;
  float* d_scale_data_fp32 = nullptr;
  if (d_scale) {
    dev_ctx.template Alloc<T>(d_scale);
    set_zero(dev_ctx, d_scale, static_cast<T>(0));
    d_scale_data = d_scale->data<T>();
    if (!std::is_same_v<XPUType, float>) {
      d_scale_data_fp32 = RAII_GUARD.alloc_l3_or_gm<float>(d_scale->numel());
    } else {
      d_scale_data_fp32 = reinterpret_cast<float*>(d_scale_data);
    }
  }
  T* d_bias_data = nullptr;
  float* d_bias_data_fp32 = nullptr;
  if (d_bias) {
    dev_ctx.template Alloc<T>(d_bias);
    set_zero(dev_ctx, d_bias, static_cast<T>(0));
    d_bias_data = d_bias->data<T>();
    if (!std::is_same_v<XPUType, float>) {
      d_bias_data_fp32 = RAII_GUARD.alloc_l3_or_gm<float>(d_bias->numel());
    } else {
      d_bias_data_fp32 = reinterpret_cast<float*>(d_bias_data);
    }
  }

  const float* scale_data = nullptr;
  if (scale_ptr) {
    if (!std::is_same_v<XPUType, float>) {
      float* scale_data_tmp =
          RAII_GUARD.alloc_l3_or_gm<float>(scale_ptr->numel());
      ret = xpu::cast<XPUType, float>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(scale_ptr->data<T>()),
          scale_data_tmp,
          scale_ptr->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "cast");
      scale_data = scale_data_tmp;
    } else {
      scale_data = scale_ptr->data<float>();
    }
  }
  const float* bias_data = nullptr;
  if (bias_ptr) {
    if (!std::is_same_v<XPUType, float>) {
      float* bias_data_tmp =
          RAII_GUARD.alloc_l3_or_gm<float>(bias_ptr->numel());
      ret = xpu::cast<XPUType, float>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(bias_ptr->data<T>()),
          bias_data_tmp,
          bias_ptr->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "cast");
      bias_data = bias_data_tmp;
    } else {
      bias_data = bias_ptr->data<float>();
    }
  }

  ret =
      xpu::group_norm_grad<XPUType>(dev_ctx.x_context(),
                                    reinterpret_cast<const XPUType*>(x_data),
                                    reinterpret_cast<const XPUType*>(y_data),
                                    reinterpret_cast<const XPUType*>(d_y_data),
                                    reinterpret_cast<XPUType*>(d_x_data),
                                    N,
                                    C,
                                    L,
                                    1,
                                    groups,
                                    epsilon,
                                    scale_data,
                                    bias_data,
                                    mean.data<float>(),
                                    var.data<float>(),
                                    d_scale_data_fp32,
                                    d_bias_data_fp32,
                                    channel_first);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "group_norm_grad");
  if (!std::is_same_v<XPUType, float>) {
    if (d_scale) {
      ret = xpu::cast<float, XPUType>(dev_ctx.x_context(),
                                      d_scale_data_fp32,
                                      reinterpret_cast<XPUType*>(d_scale_data),
                                      d_scale->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "cast");
    }

    if (d_bias) {
      ret = xpu::cast<float, XPUType>(dev_ctx.x_context(),
                                      d_bias_data_fp32,
                                      reinterpret_cast<XPUType*>(d_bias_data),
                                      d_bias->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "cast");
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(group_norm_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::GroupNormGradKernel,
                   float,
                   phi::dtype::float16) {}
