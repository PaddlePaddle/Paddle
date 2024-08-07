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

#include "paddle/phi/kernels/layer_norm_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void LayerNormGradKernel(const Context& ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& scale,
                         const paddle::optional<DenseTensor>& bias,
                         const DenseTensor& mean,
                         const DenseTensor& variance,
                         const DenseTensor& out_grad,
                         float epsilon,
                         int begin_norm_axis,
                         DenseTensor* x_grad,
                         DenseTensor* scale_grad,
                         DenseTensor* bias_grad) {
  auto x_dtype = x.dtype();
  const auto* scale_ptr = scale.get_ptr();
  const auto* bias_ptr = bias.get_ptr();
  phi::DataType scale_bias_dtype;
  if (scale_ptr != nullptr) {
    scale_bias_dtype = scale_ptr->dtype();
  } else {
    if (bias_ptr != nullptr) {
      scale_bias_dtype = bias_ptr->dtype();
    } else {
      scale_bias_dtype = x_dtype;
    }
  }

  bool is_scale_bias_same_dtype_with_x = x_dtype == scale_bias_dtype;
  if (!is_scale_bias_same_dtype_with_x) {
    PADDLE_ENFORCE_EQ(scale_bias_dtype,
                      phi::CppTypeToDataType<float>::Type(),
                      common::errors::InvalidArgument(
                          "Unsupported data type of Scale and Bias"));
  }
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto& x_dims = x.dims();
  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
  int left = static_cast<int>(matrix_dim[0]);
  int right = static_cast<int>(matrix_dim[1]);
  const auto* x_data = x.data<T>();
  const auto* out_grad_data = out_grad.data<T>();
  const auto* mean_data = mean.data<float>();
  const auto* variance_data = variance.data<float>();

  xpu::ctx_guard RAII_GUARD(ctx.x_context());

  // scale
  const float* scale_data_fp32 = nullptr;
  float* scale_grad_data_fp32 = nullptr;
  const T* scale_data_T = nullptr;
  T* scale_grad_data_T = nullptr;
  bool need_cast_scale = false;
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
    need_cast_scale = true;
    scale_grad_data_fp32 =
        scale_grad == nullptr
            ? nullptr
            : RAII_GUARD.alloc_l3_or_gm<float>(scale_ptr->numel());
  } else {
    // no need to cast
    if (is_scale_bias_same_dtype_with_x) {
      scale_data_T = scale_ptr->data<T>();
      scale_grad_data_T =
          scale_grad == nullptr ? nullptr : ctx.template Alloc<T>(scale_grad);
    } else {
      scale_data_fp32 = scale_ptr->data<float>();
      scale_grad_data_fp32 = scale_grad == nullptr
                                 ? nullptr
                                 : ctx.template Alloc<float>(scale_grad);
    }
  }

  // bias
  float* bias_grad_data_fp32 = nullptr;
  T* bias_grad_data_T = nullptr;
  bool need_cast_bias = false;
  if (bias_ptr == nullptr) {
    // no bias, do nothing
  } else if (bias_ptr->dtype() ==
             phi::CppTypeToDataType<phi::dtype::float16>::Type()) {
    need_cast_bias = true;
    bias_grad_data_fp32 =
        bias_grad == nullptr
            ? nullptr
            : RAII_GUARD.alloc_l3_or_gm<float>(bias_ptr->numel());
  } else {
    // no need to cast
    if (is_scale_bias_same_dtype_with_x) {
      bias_grad_data_T =
          bias_grad == nullptr ? nullptr : ctx.template Alloc<T>(bias_grad);
    } else {
      bias_grad_data_fp32 =
          bias_grad == nullptr ? nullptr : ctx.template Alloc<float>(bias_grad);
    }
  }

  auto* x_grad_data =
      (x_grad == nullptr ? nullptr : ctx.template Alloc<T>(x_grad));

  if (!is_scale_bias_same_dtype_with_x) {
    int r =
        xpu::layer_norm_grad(ctx.x_context(),
                             reinterpret_cast<const XPUType*>(x_data),
                             reinterpret_cast<const XPUType*>(out_grad_data),
                             reinterpret_cast<XPUType*>(x_grad_data),
                             left,
                             right,
                             epsilon,
                             scale_data_fp32,
                             mean_data,
                             variance_data,
                             scale_grad_data_fp32,
                             bias_grad_data_fp32);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
  } else {
    int r =
        xpu::layer_norm_grad(ctx.x_context(),
                             reinterpret_cast<const XPUType*>(x_data),
                             reinterpret_cast<const XPUType*>(out_grad_data),
                             reinterpret_cast<XPUType*>(x_grad_data),
                             left,
                             right,
                             epsilon,
                             reinterpret_cast<const XPUType*>(scale_data_T),
                             mean_data,
                             variance_data,
                             reinterpret_cast<XPUType*>(scale_grad_data_T),
                             reinterpret_cast<XPUType*>(bias_grad_data_T));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
  }

  if (need_cast_scale) {
    int r = xpu::cast<float, XPUType>(
        ctx.x_context(),
        scale_grad_data_fp32,
        reinterpret_cast<XPUType*>(ctx.template Alloc<T>(scale_grad)),
        scale.get_ptr()->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  }
  if (need_cast_bias) {
    int r = xpu::cast<float, XPUType>(
        ctx.x_context(),
        bias_grad_data_fp32,
        reinterpret_cast<XPUType*>(ctx.template Alloc<T>(bias_grad)),
        bias.get_ptr()->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(layer_norm_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::LayerNormGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
