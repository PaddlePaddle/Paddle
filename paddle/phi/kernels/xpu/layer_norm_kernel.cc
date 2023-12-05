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

#include "paddle/phi/kernels/layer_norm_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void LayerNormKernel(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int begin_norm_axis,
                     DenseTensor* out,
                     DenseTensor* mean,
                     DenseTensor* variance) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto& x_dims = x.dims();
  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
  int left = static_cast<int>(matrix_dim[0]);
  int right = static_cast<int>(matrix_dim[1]);
  const auto* x_data = x.data<T>();

  xpu::ctx_guard RAII_GUARD(ctx.x_context());

  // scale
  const float* scale_data_fp32 = nullptr;
  const auto* scale_ptr = scale.get_ptr();
  if (scale_ptr == nullptr) {
    // no scale, do nothing
  } else if (scale_ptr->dtype() ==
             phi::CppTypeToDataType<phi::dtype::float16>::Type()) {
    float* scale_data_temp =
        RAII_GUARD.alloc_l3_or_gm<float>(scale_ptr->numel());
    int r = xpu::cast<XPUType, float>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(scale_ptr->data<T>()),
        scale_data_temp,
        scale_ptr->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    scale_data_fp32 = scale_data_temp;
  } else {
    // no need to cast
    scale_data_fp32 = scale_ptr->data<float>();
  }

  // bias
  const float* bias_data_fp32 = nullptr;
  const auto* bias_ptr = bias.get_ptr();
  if (bias_ptr == nullptr) {
    // no bias, do nothing
  } else if (bias_ptr->dtype() ==
             phi::CppTypeToDataType<phi::dtype::float16>::Type()) {
    float* bias_data_temp = RAII_GUARD.alloc_l3_or_gm<float>(bias_ptr->numel());
    int r = xpu::cast<XPUType, float>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(bias_ptr->data<T>()),
        bias_data_temp,
        bias_ptr->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    bias_data_fp32 = bias_data_temp;
  } else {
    // no need to cast
    bias_data_fp32 = bias_ptr->data<float>();
  }

  auto* out_data = ctx.template Alloc<T>(out);
  auto* mean_data = ctx.template Alloc<float>(mean);
  auto* variance_data = ctx.template Alloc<float>(variance);

  // int layer_norm(Context* ctx, const T* x, T* y, int64_t m, int64_t n, float
  // eps, const float* scale, const float* bias, float* mean, float* var, bool
  // is_rstd = false);
  int r = xpu::layer_norm(ctx.x_context(),
                          reinterpret_cast<const XPUType*>(x_data),
                          reinterpret_cast<XPUType*>(out_data),
                          left,
                          right,
                          epsilon,
                          scale_data_fp32,
                          bias_data_fp32,
                          mean_data,
                          variance_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
}
}  // namespace phi

PD_REGISTER_KERNEL(layer_norm,
                   XPU,
                   ALL_LAYOUT,
                   phi::LayerNormKernel,
                   float,
                   phi::dtype::float16) {}
