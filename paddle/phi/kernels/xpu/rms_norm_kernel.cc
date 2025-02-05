/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Original OneFlow copyright notice:

/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/layer_norm.cuh
// The following code modified from OneFlow's implementation, and change to use
// single Pass algorithm. Support Int8 quant, dequant Load/Store implementation.

#include "paddle/phi/kernels/rms_norm_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RmsNormKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const paddle::optional<DenseTensor>& bias,
                   const paddle::optional<DenseTensor>& residual,
                   const DenseTensor& norm_weight,
                   const paddle::optional<DenseTensor>& norm_bias,
                   const float epsilon,
                   const int begin_norm_axis,
                   const float quant_scale,
                   const int quant_round_type,
                   const float quant_max_bound,
                   const float quant_min_bound,
                   DenseTensor* out,
                   DenseTensor* residual_out,
                   DenseTensor* inv_var) {
  if (quant_scale > 0.0f) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Quantization is not supported in XPU rms_norm yet"));
  }
  PADDLE_ENFORCE_EQ(
      begin_norm_axis > 0 && begin_norm_axis <= x.dims().size(),
      true,
      common::errors::InvalidArgument("begin_norm_axis should be in range [1, "
                                      "%d], but received begin_norm_axis = "
                                      "%d",
                                      x.dims().size(),
                                      begin_norm_axis));
  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  const T* x_data = x.data<T>();
  const T* norm_weight_data = norm_weight.data<T>();
  const T* norm_bias_data = norm_bias ? norm_bias.get().data<T>() : nullptr;

  dev_ctx.template Alloc<T>(out);
  T* out_data = out->data<T>();
  float* inv_var_data = nullptr;
  if (inv_var != nullptr && std::getenv("XPU_RMS_NORM_INFER_OPT") == nullptr) {
    dev_ctx.template Alloc<float>(inv_var);
    inv_var_data = inv_var->data<float>();
  }

  int32_t rows = 1;
  int32_t cols = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    rows *= x.dims()[i];
  }

  for (int i = begin_norm_axis; i < x.dims().size(); i++) {
    cols *= x.dims()[i];
  }

  PADDLE_ENFORCE_EQ(
      cols,
      norm_weight.dims()[0],
      common::errors::InvalidArgument(
          "The product from begin_norm_axis to the last axis of input tensor "
          "x, "
          "i.e., cols(%d)"
          "must be equal to the norm_weight tensor's dimension(%d).",
          cols,
          norm_weight.dims()[0]));
  if (norm_bias) {
    PADDLE_ENFORCE_EQ(
        cols,
        norm_bias.get().dims()[0],
        common::errors::InvalidArgument(
            "The product from begin_norm_axis to the last axis of input tensor "
            "x, "
            "i.e., cols(%d)"
            "must be equal to the norm_bias tensor's dimension(%d).",
            cols,
            norm_bias.get().dims()[0]));
  }

  if (residual) {
    // Do RMSNorm(bias_add + residual + x)
    T* residual_out_data = dev_ctx.template Alloc<T>(residual_out);
    const T* residual_data = residual.get().data<T>();
    const T* bias_data = bias ? bias.get().data<T>() : nullptr;
    int r = xpu::add(dev_ctx.x_context(),
                     reinterpret_cast<const XPUType*>(x_data),
                     reinterpret_cast<const XPUType*>(residual_data),
                     reinterpret_cast<XPUType*>(residual_out_data),
                     x.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
    if (bias_data) {
      r = xpu::broadcast_add(dev_ctx.x_context(),
                             reinterpret_cast<XPUType*>(residual_out_data),
                             reinterpret_cast<const XPUType*>(bias_data),
                             reinterpret_cast<XPUType*>(residual_out_data),
                             {rows, cols},
                             {cols});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
    }
    r = xpu::rms_layer_norm<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<XPUType*>(residual_out_data),
        reinterpret_cast<XPUType*>(out_data),
        rows,
        cols,
        epsilon,
        reinterpret_cast<const XPUType*>(norm_weight_data),
        reinterpret_cast<const XPUType*>(norm_bias_data),
        inv_var_data,
        false);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "rms_layer_norm");
  } else {
    int r = xpu::rms_layer_norm<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x_data),
        reinterpret_cast<XPUType*>(out_data),
        rows,
        cols,
        epsilon,
        reinterpret_cast<const XPUType*>(norm_weight_data),
        reinterpret_cast<const XPUType*>(norm_bias_data),
        inv_var_data,
        false);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "rms_layer_norm");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(rms_norm,
                   XPU,
                   ALL_LAYOUT,
                   phi::RmsNormKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
