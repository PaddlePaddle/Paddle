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

#include "paddle/phi/kernels/fused_layernorm_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

namespace fusion {

template <typename T, typename Context>
void FusedLayerNormKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const optional<DenseTensor>& bias,
                          const optional<DenseTensor>& residual,
                          const optional<DenseTensor>& norm_weight,
                          const optional<DenseTensor>& norm_bias,
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
  auto xpu_ctx = static_cast<const XPUContext*>(&dev_ctx);
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

  PADDLE_ENFORCE_EQ(
      out->dtype() != phi::DataType::FLOAT8_E4M3FN,
      true,
      common::errors::Unimplemented(
          "XPU fused_bias_residual_layernorm does not support FLOAT8_E4M3FN "
          "quantized output yet."));
  const bool quant_int8 = out->dtype() == phi::DataType::INT8;
  DenseTensor fp_out;
  DenseTensor quant_input;
  DenseTensor* ln_out = out;
  if (quant_int8) {
    fp_out.Resize(out->dims());
    quant_input.Resize(out->dims());
    dev_ctx.template Alloc<float>(&fp_out);
    dev_ctx.template Alloc<T>(&quant_input);
    ln_out = &quant_input;
  } else {
    dev_ctx.template Alloc<T>(out);
  }
  dev_ctx.template Alloc<float>(mean);
  dev_ctx.template Alloc<float>(variance);

  if (m * n == 0) {
    if (quant_int8) {
      dev_ctx.template Alloc<int8_t>(out);
    }
    if (residual) {
      dev_ctx.template Alloc<T>(residual_out);
    }
    return;
  }

  DenseTensor residual_alpha_tmp;
  residual_alpha_tmp.Resize({1});

  DenseTensor residual_alpha_ptr;
  residual_alpha_ptr.Resize({1});

  dev_ctx.template Alloc<float>(&residual_alpha_tmp);
  dev_ctx.template Alloc<T>(&residual_alpha_ptr);

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

  if (!norm_weight && !norm_bias) {
    if (residual) {
      r = baidu::xpu::api::broadcast_mul(
          xpu_ctx->x_context(),
          reinterpret_cast<const XPUType*>(residual.get().data<T>()),
          reinterpret_cast<XPUType*>(residual_alpha_ptr.data<T>()),
          reinterpret_cast<XPUType*>(ln_out->data<T>()),
          {m, n},
          {1});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");
      if (bias) {
        r = baidu::xpu::api::broadcast_add(
            xpu_ctx->x_context(),
            reinterpret_cast<XPUType*>(ln_out->data<T>()),
            reinterpret_cast<const XPUType*>(bias.get().data<T>()),
            reinterpret_cast<XPUType*>(ln_out->data<T>()),
            {m, n},
            {n});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
      }
      r = baidu::xpu::api::add(xpu_ctx->x_context(),
                               reinterpret_cast<XPUType*>(ln_out->data<T>()),
                               reinterpret_cast<const XPUType*>(x.data<T>()),
                               reinterpret_cast<XPUType*>(ln_out->data<T>()),
                               m * n);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
      dev_ctx.template Alloc<T>(residual_out);
      r = baidu::xpu::api::copy(
          xpu_ctx->x_context(),
          reinterpret_cast<XPUType*>(ln_out->data<T>()),
          reinterpret_cast<XPUType*>(residual_out->data<T>()),
          m * n);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    } else {
      if (bias) {
        r = baidu::xpu::api::broadcast_add(
            xpu_ctx->x_context(),
            reinterpret_cast<const XPUType*>(x.data<T>()),
            reinterpret_cast<const XPUType*>(bias.get().data<T>()),
            reinterpret_cast<XPUType*>(ln_out->data<T>()),
            {m, n},
            {n});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
      } else {
        r = baidu::xpu::api::copy(xpu_ctx->x_context(),
                                  reinterpret_cast<const XPUType*>(x.data<T>()),
                                  reinterpret_cast<XPUType*>(ln_out->data<T>()),
                                  m * n);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
      }
    }
    if (!quant_int8 || quant_scale <= 0.0f) {
      return;
    }
    r = baidu::xpu::api::cast<XPUType, float>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType*>(ln_out->data<T>()),
        fp_out.data<float>(),
        m * n);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  } else {
    auto x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
    if (bias) {
      DenseTensor x_tmp;
      x_tmp.Resize(x.dims());
      dev_ctx.template Alloc<T>(&x_tmp);
      r = baidu::xpu::api::broadcast_add(
          xpu_ctx->x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<const XPUType*>(bias.get().data<T>()),
          reinterpret_cast<XPUType*>(x_tmp.data<T>()),
          {m, n},
          {n});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
      x_ptr = reinterpret_cast<XPUType*>(x_tmp.data<T>());
    }
    if (residual) {
      if (std::is_same<T, phi::bfloat16>::value) {
        PD_THROW("NOT supported quant bfloat16. ");
      }
      dev_ctx.template Alloc<T>(residual_out);
      DenseTensor residual_tmp;
      residual_tmp.Resize(residual.get().dims());
      dev_ctx.template Alloc<T>(&residual_tmp);
      r = baidu::xpu::api::broadcast_mul(
          xpu_ctx->x_context(),
          reinterpret_cast<const XPUType*>(residual.get().data<T>()),
          reinterpret_cast<XPUType*>(residual_alpha_ptr.data<T>()),
          reinterpret_cast<XPUType*>(residual_tmp.data<T>()),
          {m, n},
          {1});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

      if (quant_int8) {
        DenseTensor ln_input;
        ln_input.Resize(x.dims());
        dev_ctx.template Alloc<T>(&ln_input);
        r = baidu::xpu::api::add(
            xpu_ctx->x_context(),
            x_ptr,
            reinterpret_cast<const XPUType*>(residual_tmp.data<T>()),
            reinterpret_cast<XPUType*>(ln_input.data<T>()),
            m * n);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
        r = baidu::xpu::api::copy(
            xpu_ctx->x_context(),
            reinterpret_cast<const XPUType*>(ln_input.data<T>()),
            reinterpret_cast<XPUType*>(residual_out->data<T>()),
            m * n);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
        DenseTensor ln_out_tmp;
        ln_out_tmp.Resize(out->dims());
        dev_ctx.template Alloc<T>(&ln_out_tmp);
        r = baidu::xpu::api::layer_norm(
            xpu_ctx->x_context(),
            reinterpret_cast<const XPUType*>(ln_input.data<T>()),
            reinterpret_cast<XPUType*>(ln_out_tmp.data<T>()),
            m,
            n,
            epsilon,
            norm_weight.get().data<float>(),
            norm_bias.get().data<float>(),
            mean->data<float>(),
            variance->data<float>(),
            true);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
        DenseTensor ln_input_fp;
        ln_input_fp.Resize(x.dims());
        dev_ctx.template Alloc<float>(&ln_input_fp);
        r = baidu::xpu::api::cast<XPUType, float>(
            xpu_ctx->x_context(),
            reinterpret_cast<const XPUType*>(ln_input.data<T>()),
            ln_input_fp.data<float>(),
            m * n);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
        DenseTensor quant_mean;
        DenseTensor quant_variance;
        quant_mean.Resize(mean->dims());
        quant_variance.Resize(variance->dims());
        dev_ctx.template Alloc<float>(&quant_mean);
        dev_ctx.template Alloc<float>(&quant_variance);
        r = baidu::xpu::api::layer_norm(xpu_ctx->x_context(),
                                        ln_input_fp.data<float>(),
                                        fp_out.data<float>(),
                                        m,
                                        n,
                                        epsilon,
                                        norm_weight.get().data<float>(),
                                        norm_bias.get().data<float>(),
                                        quant_mean.data<float>(),
                                        quant_variance.data<float>(),
                                        true);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
      } else {
        r = baidu::xpu::api::add_layer_norm_fusion(
            xpu_ctx->x_context(),
            x_ptr,
            reinterpret_cast<const XPUType*>(residual_tmp.data<T>()),
            reinterpret_cast<XPUType*>(ln_out->data<T>()),
            m,
            n,
            epsilon,
            norm_weight.get().data<float>(),
            norm_bias.get().data<float>(),
            mean->data<float>(),
            variance->data<float>(),
            reinterpret_cast<XPUType*>(residual_out->data<T>()));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "add_layer_norm_fusion");
      }
    } else {
      if (quant_int8) {
        DenseTensor ln_out_tmp;
        ln_out_tmp.Resize(out->dims());
        dev_ctx.template Alloc<T>(&ln_out_tmp);
        r = baidu::xpu::api::layer_norm(
            xpu_ctx->x_context(),
            x_ptr,
            reinterpret_cast<XPUType*>(ln_out_tmp.data<T>()),
            m,
            n,
            epsilon,
            norm_weight.get().data<float>(),
            norm_bias.get().data<float>(),
            mean->data<float>(),
            variance->data<float>(),
            true);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
        DenseTensor x_fp;
        x_fp.Resize(x.dims());
        dev_ctx.template Alloc<float>(&x_fp);
        r = baidu::xpu::api::cast<XPUType, float>(
            xpu_ctx->x_context(), x_ptr, x_fp.data<float>(), m * n);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
        DenseTensor quant_mean;
        DenseTensor quant_variance;
        quant_mean.Resize(mean->dims());
        quant_variance.Resize(variance->dims());
        dev_ctx.template Alloc<float>(&quant_mean);
        dev_ctx.template Alloc<float>(&quant_variance);
        r = baidu::xpu::api::layer_norm(xpu_ctx->x_context(),
                                        x_fp.data<float>(),
                                        fp_out.data<float>(),
                                        m,
                                        n,
                                        epsilon,
                                        norm_weight.get().data<float>(),
                                        norm_bias.get().data<float>(),
                                        quant_mean.data<float>(),
                                        quant_variance.data<float>(),
                                        true);
      } else {
        r = baidu::xpu::api::layer_norm(
            xpu_ctx->x_context(),
            x_ptr,
            reinterpret_cast<XPUType*>(ln_out->data<T>()),
            m,
            n,
            epsilon,
            norm_weight.get().data<float>(),
            norm_bias.get().data<float>(),
            mean->data<float>(),
            variance->data<float>(),
            true);
      }
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
    }
    if (quant_scale <= 0.0f) {
      return;
    }
  }

  if (quant_scale > 0.0f) {
    PADDLE_ENFORCE_EQ(quant_int8,
                      true,
                      common::errors::Unimplemented(
                          "XPU fused_bias_residual_layernorm only supports "
                          "INT8 quantized output."));
    PADDLE_ENFORCE_EQ(quant_round_type == 1,
                      true,
                      common::errors::InvalidArgument(
                          "XPU fused_bias_residual_layernorm quantized output "
                          "only supports quant_round_type = 1, but got %d.",
                          quant_round_type));
    DenseTensor quant_tmp;
    quant_tmp.Resize(out->dims());
    dev_ctx.template Alloc<float>(&quant_tmp);
    r = baidu::xpu::api::scale(xpu_ctx->x_context(),
                               fp_out.data<float>(),
                               quant_tmp.data<float>(),
                               m * n,
                               false,
                               quant_max_bound * quant_scale,
                               0.f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    r = baidu::xpu::api::round(xpu_ctx->x_context(),
                               quant_tmp.data<float>(),
                               quant_tmp.data<float>(),
                               m * n,
                               0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "round");
    r = baidu::xpu::api::clip(xpu_ctx->x_context(),
                              quant_tmp.data<float>(),
                              quant_tmp.data<float>(),
                              m * n,
                              quant_min_bound,
                              quant_max_bound);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip");
    dev_ctx.template Alloc<int8_t>(out);
    r = baidu::xpu::api::cast<float, int8_t>(xpu_ctx->x_context(),
                                             quant_tmp.data<float>(),
                                             out->data<int8_t>(),
                                             m * n);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  }
}

}  // namespace fusion

}  // namespace phi

PD_REGISTER_KERNEL(fused_bias_residual_layernorm,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedLayerNormKernel,
                   float,
                   phi::bfloat16,
                   phi::float16) {
  kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
}
