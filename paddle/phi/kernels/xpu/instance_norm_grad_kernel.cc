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

#include "paddle/phi/kernels/instance_norm_grad_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"

namespace phi {

template <typename T, typename Context>
void InstanceNormGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const paddle::optional<DenseTensor>& scale,
                            const DenseTensor& saved_mean,
                            const DenseTensor& saved_variance,
                            const DenseTensor& d_y,
                            float epsilon,
                            DenseTensor* d_x,
                            DenseTensor* d_scale,
                            DenseTensor* d_bias) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto& x_dims = x.dims();
  int N, C, H, W, D;
  funcs::ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
  PADDLE_ENFORCE_EQ(
      x_dims.size() <= 5 && D == 1,
      true,
      common::errors::InvalidArgument(
          "The size of input's dimensions should be less equal than 5",
          "and the dimension of D should be equal to 1",
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  dev_ctx.template Alloc<T>(d_x);
  T* d_scale_data = nullptr;
  T* d_bias_data = nullptr;
  if (d_scale && d_bias) {
    dev_ctx.template Alloc<float>(d_scale);
    dev_ctx.template Alloc<float>(d_bias);
    d_scale_data = d_scale->data<float>();
    d_bias_data = d_bias->data<float>();
  }

  const auto scale_ptr = scale.get_ptr();
  if (scale_ptr) {
    PADDLE_ENFORCE_EQ(
        scale_ptr->dims().size(),
        1UL,
        common::errors::InvalidArgument(
            "The `shape` in InstanceNormOp is invalid: "
            "the size of scale's dimensions must be equal to 1. But "
            "received: the size of scale's dimensions"
            "is [%d]",
            scale_ptr->dims().size()));
    PADDLE_ENFORCE_EQ(scale_ptr->dims()[0],
                      C,
                      common::errors::InvalidArgument(
                          "The `shape` in InstanceNormOp is invalid: "
                          "the first dimension of scale must be equal to "
                          "Channels([%d]). But received: "
                          "the first dimension of scale is [%d],"
                          "the dimensions of scale is [%s], ",
                          C,
                          scale_ptr->dims()[0],
                          scale_ptr->dims()));
  }

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  float* scale_ptr_data_tmp;
  int r;
  if (!scale_ptr) {
    scale_ptr_data_tmp = RAII_GUARD.alloc_l3_or_gm<float>(C);
    r = xpu::constant(dev_ctx.x_context(),
                      reinterpret_cast<float*>(scale_ptr_data_tmp),
                      C,
                      static_cast<float>(1));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  }
  auto scale_ptr_data =
      scale_ptr ? scale_ptr->data<float>() : scale_ptr_data_tmp;

  if ((H * W * D) == 1) {
    r = xpu::copy(dev_ctx.x_context(),
                  reinterpret_cast<const XPUType*>(d_y.data<T>()),
                  reinterpret_cast<XPUType*>(d_x->data<T>()),
                  d_y.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    r = xpu::constant(dev_ctx.x_context(),
                      reinterpret_cast<float*>(d_scale),
                      C,
                      static_cast<float>(0));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    r = xpu::constant(dev_ctx.x_context(),
                      reinterpret_cast<float*>(d_bias),
                      C,
                      static_cast<float>(0));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    return;
  }
  auto d_x_data =
      d_x ? d_x->data<T>() : RAII_GUARD.alloc_l3_or_gm<T>(x.numel());
  r = xpu::instance_norm_grad(dev_ctx.x_context(),
                              reinterpret_cast<const XPUType*>(x.data<T>()),
                              reinterpret_cast<const XPUType*>(d_y.data<T>()),
                              reinterpret_cast<XPUType*>(d_x_data),
                              scale_ptr_data,
                              saved_mean.data<float>(),
                              saved_variance.data<float>(),
                              d_scale_data,
                              d_bias_data,
                              N,
                              C,
                              H,
                              W,
                              epsilon,
                              true);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "instance_norm_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    instance_norm_grad, XPU, ALL_LAYOUT, phi::InstanceNormGradKernel, float) {}
