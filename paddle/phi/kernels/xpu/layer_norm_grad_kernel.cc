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
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T, typename TW, typename Context>  // TW for scale and bias
void LayerNormGradImpl(const Context& dev_ctx,
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
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    if (scale_grad)
      phi::Full<T, Context>(
          dev_ctx,
          phi::IntArray(common::vectorize(scale_grad->dims())),
          0,
          scale_grad);
    if (bias_grad)
      phi::Full<T, Context>(dev_ctx,
                            phi::IntArray(common::vectorize(bias_grad->dims())),
                            0,
                            bias_grad);
    return;
  }
  const auto* scale_ptr = scale.get_ptr();
  using XPUType = typename XPUTypeTrait<T>::Type;
  using XPUTypeTW = typename XPUTypeTrait<TW>::Type;
  const auto& x_dims = x.dims();
  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t left = matrix_dim[0];
  int64_t right = matrix_dim[1];
  const auto* x_data = x.data<T>();
  const auto* out_grad_data = out_grad.data<T>();
  const auto* mean_data = mean.data<float>();
  const auto* variance_data = variance.data<float>();

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  T* x_grad_data = nullptr;
  const TW* scale_data = nullptr;
  TW* scale_grad_data = nullptr;
  TW* bias_grad_data = nullptr;
  if (x_grad != nullptr) {
    dev_ctx.template Alloc<T>(x_grad);
    x_grad_data = x_grad->data<T>();
  }
  if (scale_ptr != nullptr) {
    scale_data = scale_ptr->data<TW>();
    if (scale_grad != nullptr) {
      dev_ctx.template Alloc<TW>(scale_grad);
      scale_grad_data = scale_grad->data<TW>();
    }
  }
  if (bias_grad != nullptr) {
    dev_ctx.template Alloc<TW>(bias_grad);
    bias_grad_data = bias_grad->data<TW>();
  }
  int r = xpu::layer_norm_grad(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(x_data),
                               reinterpret_cast<const XPUType*>(out_grad_data),
                               reinterpret_cast<XPUType*>(x_grad_data),
                               left,
                               right,
                               epsilon,
                               reinterpret_cast<const XPUTypeTW*>(scale_data),
                               mean_data,
                               variance_data,
                               reinterpret_cast<XPUTypeTW*>(scale_grad_data),
                               reinterpret_cast<XPUTypeTW*>(bias_grad_data));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
}

template <typename T, typename Context>
void LayerNormGradKernel(const Context& dev_ctx,
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

  bool is_scale_bias_same_dtype_with_x = (x_dtype == scale_bias_dtype);
  if (!is_scale_bias_same_dtype_with_x) {
    PADDLE_ENFORCE_EQ(scale_bias_dtype,
                      phi::CppTypeToDataType<float>::Type(),
                      common::errors::InvalidArgument(
                          "Unsupported data type of Scale and Bias"));
  }

  if (is_scale_bias_same_dtype_with_x) {
    LayerNormGradImpl<T, T, Context>(dev_ctx,
                                     x,
                                     scale,
                                     bias,
                                     mean,
                                     variance,
                                     out_grad,
                                     epsilon,
                                     begin_norm_axis,
                                     x_grad,
                                     scale_grad,
                                     bias_grad);
  } else {
    LayerNormGradImpl<T, float, Context>(dev_ctx,
                                         x,
                                         scale,
                                         bias,
                                         mean,
                                         variance,
                                         out_grad,
                                         epsilon,
                                         begin_norm_axis,
                                         x_grad,
                                         scale_grad,
                                         bias_grad);
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
