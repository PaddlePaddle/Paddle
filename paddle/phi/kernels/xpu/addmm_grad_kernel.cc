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

#include "paddle/phi/kernels/addmm_grad_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

template <typename T, typename Context>
void AddmmGradKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out_grad,
                     float alpha,
                     float beta,
                     DenseTensor* input_grad,
                     DenseTensor* x_grad,
                     DenseTensor* y_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  xpu::Context* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  int r;

  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    XPUType* input_grad_ptr = reinterpret_cast<XPUType*>(input_grad->data<T>());
    r = xpu::constant(xpu_ctx, input_grad_ptr, input.numel(), (XPUType)(beta));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    if (input_grad->dims().size() == 1 && out_grad.dims()[0] > 1) {
      r = xpu::scale<XPUType>(xpu_ctx,
                              input_grad_ptr,
                              input_grad_ptr,
                              input_grad->numel(),
                              true,
                              static_cast<float>(out_grad.dims()[0]),
                              0.f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }
  }
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
  }
  if (y_grad) {
    dev_ctx.template Alloc<T>(y_grad);
  }

  const XPUType* out_grad_ptr =
      reinterpret_cast<const XPUType*>(out_grad.data<T>());
  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y.data<T>());

  XpuFcInfo info_forward;
  GetFCInfo(x.dims(), y.dims(), false, false, &info_forward);
  // begin calculate
  const XPUType* a_1 = nullptr;
  const XPUType* b_1 = nullptr;
  const XPUType* a_2 = nullptr;
  const XPUType* b_2 = nullptr;
  XPUType* c_1 = reinterpret_cast<XPUType*>(x_grad->data<T>());
  XPUType* c_2 = reinterpret_cast<XPUType*>(y_grad->data<T>());

  if (x_grad && info_forward.is_x_need_broadcast) {
    c_1 = RAII_GUARD.alloc_l3_or_gm<XPUType>(info_forward.bs * info_forward.m *
                                             info_forward.k);
    PADDLE_ENFORCE_XDNN_NOT_NULL(c_1);
  }

  if (y_grad && info_forward.is_y_need_broadcast) {
    c_2 = RAII_GUARD.alloc_l3_or_gm<XPUType>(info_forward.bs * info_forward.k *
                                             info_forward.n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(c_2);
  }

  XpuFcInfo info_x_grad;
  XpuFcInfo info_y_grad;
  std::tuple<XpuFcInfo,
             XpuFcInfo,
             const XPUType*,
             const XPUType*,
             const XPUType*,
             const XPUType*>
      fc_info = MatmulGradFcInfo(xpu_ctx,
                                 &RAII_GUARD,
                                 info_forward,
                                 false,
                                 false,
                                 x_ptr,
                                 y_ptr,
                                 out_grad_ptr);
  std::tie(info_x_grad, info_y_grad, a_1, b_1, a_2, b_2) = fc_info;
  if (x_grad) {
    MatMulXPUFunction<XPUType>(xpu_ctx, a_1, b_1, c_1, info_x_grad, alpha, 0.f);
    if (info_forward.is_x_need_broadcast) {
      r = xpu::reduce_sum<XPUType>(
          xpu_ctx,
          c_1,
          reinterpret_cast<XPUType*>(x_grad->data<T>()),
          {info_forward.bs, info_forward.m, info_forward.k},
          {0});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
    }
  }
  if (y_grad) {
    MatMulXPUFunction<XPUType>(xpu_ctx, a_2, b_2, c_2, info_y_grad, alpha, 0.f);
    if (info_forward.is_y_need_broadcast) {
      r = xpu::reduce_sum<XPUType>(
          xpu_ctx,
          c_2,
          reinterpret_cast<XPUType*>(y_grad->data<T>()),
          {info_forward.bs, info_forward.k, info_forward.n},
          {0});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(addmm_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::AddmmGradKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
