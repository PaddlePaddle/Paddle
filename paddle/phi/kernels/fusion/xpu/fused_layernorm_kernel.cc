// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

namespace fusion {

template <typename T, typename Context>
void FusedLayerNormKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const paddle::optional<DenseTensor>& bias,
                          const paddle::optional<DenseTensor>& residual,
                          const paddle::optional<DenseTensor>& norm_weight,
                          const paddle::optional<DenseTensor>& norm_bias,
                          const float epsilon,
                          const float residual_alpha,
                          const int begin_norm_axis,
                          const float quant_scale,
                          const int quant_round_type,
                          const float quant_max_bound,
                          const float quant_min_bound,
                          DenseTensor* out,
                          DenseTensor* residual_out,
                          DenseTensor* mean,
                          DenseTensor* variance) {
  int r = xpu::SUCCESS;
  auto xpu_ctx = static_cast<const phi::XPUContext*>(&dev_ctx);
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto x_shape = x.dims();
  int m = 1;
  int n = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    m *= x_shape[i];
  }
  for (int i = begin_norm_axis; i < x_shape.size(); i++) {
    n *= x_shape[i];
  }

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<float>(mean);
  dev_ctx.template Alloc<float>(variance);

  DenseTensor residual_alpha_tmp;
  residual_alpha_tmp.Resize({1});

  DenseTensor residual_alpha_ptr;
  residual_alpha_ptr.Resize({1});

  dev_ctx.template Alloc<float>(&residual_alpha_tmp);
  dev_ctx.template Alloc<T>(&residual_alpha_ptr);
  r = baidu::xpu::api::constant(xpu_ctx->x_context(),
                                reinterpret_cast<XPUType*>(out->data<T>()),
                                out->numel(),
                                static_cast<XPUType>(0.f));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

  r = baidu::xpu::api::constant(xpu_ctx->x_context(),
                                residual_alpha_tmp.data<float>(),
                                1,
                                residual_alpha);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

  r = baidu::xpu::api::cast(
      xpu_ctx->x_context(),
      residual_alpha_tmp.data<float>(),
      reinterpret_cast<XPUType*>(residual_alpha_ptr.data<T>()),
      1);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");

  if (residual) {
    dev_ctx.template Alloc<T>(residual_out);
    r = baidu::xpu::api::broadcast_mul(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType*>(residual.get().data<T>()),
        reinterpret_cast<XPUType*>(residual_alpha_ptr.data<T>()),
        reinterpret_cast<XPUType*>(const_cast<T*>(residual.get().data<T>())),
        {m, n},
        {1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");
  }

  if (!norm_weight && !norm_bias) {
    if (bias) {
      r = baidu::xpu::api::broadcast_add(
          xpu_ctx->x_context(),
          reinterpret_cast<XPUType*>(out->data<T>()),
          reinterpret_cast<const XPUType*>(bias.get().data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          {m, n},
          {n});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
    }
    if (residual) {
      r = baidu::xpu::api::add(
          xpu_ctx->x_context(),
          reinterpret_cast<XPUType*>(out->data<T>()),
          reinterpret_cast<const XPUType*>(residual.get().data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          m * n);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
    }

    r = baidu::xpu::api::add(xpu_ctx->x_context(),
                             reinterpret_cast<XPUType*>(out->data<T>()),
                             reinterpret_cast<const XPUType*>(x.data<T>()),
                             reinterpret_cast<XPUType*>(out->data<T>()),
                             m * n);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
    return;
  } else {
    if (bias) {
      r = baidu::xpu::api::broadcast_add(
          xpu_ctx->x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<const XPUType*>(bias.get().data<T>()),
          reinterpret_cast<XPUType*>(const_cast<T*>((x.data<T>()))),
          {m, n},
          {n});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
    }
    if (residual) {
      r = baidu::xpu::api::add_layer_norm_fusion(
          xpu_ctx->x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<const XPUType*>(residual.get().data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          m,
          n,
          epsilon,
          norm_weight.get().data<float>(),
          norm_bias.get().data<float>(),
          mean->data<float>(),
          variance->data<float>(),
          reinterpret_cast<XPUType*>(residual_out->data<T>()));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "add_layer_norm_fusion");
    } else {
      r = baidu::xpu::api::layer_norm(
          xpu_ctx->x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          m,
          n,
          epsilon,
          norm_weight.get().data<float>(),
          norm_bias.get().data<float>(),
          mean->data<float>(),
          variance->data<float>());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
    }
    if (quant_scale > 0.0f) {
      PD_THROW("NOT supported quant int8. ");
    } else {
      return;
    }
  }
}

}  // namespace fusion

}  // namespace phi

PD_REGISTER_KERNEL(fused_bias_residual_layernorm,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedLayerNormKernel,
                   float,
                   phi::dtype::float16) {}
