/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/cross_entropy_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace phi {

template <typename T, typename Context>
void CrossEntropyWithSoftmaxGradKernel(const Context& dev_ctx,
                                       const DenseTensor& labels,
                                       const DenseTensor& softmax,
                                       const DenseTensor& loss_grad,
                                       bool soft_label,
                                       bool use_softmax,
                                       bool numeric_stable_mode,
                                       int ignore_index,
                                       int axis_in,
                                       DenseTensor* logit_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(logit_grad);

  const int rank = logit_grad->dims().size();
  const int axis = phi::funcs::CanonicalAxis(axis_in, rank);
  const int n = phi::funcs::SizeToAxis(axis, logit_grad->dims());
  const int d = phi::funcs::SizeFromAxis(axis, logit_grad->dims());

  int r = XPU_SUCCESS;

  if (axis == rank - 1) {
    if (soft_label) {
      r = xpu::soft_softmax_with_cross_entropy_grad<XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(loss_grad.data<T>()),
          reinterpret_cast<const XPUType*>(labels.data<T>()),
          reinterpret_cast<const XPUType*>(softmax.data<T>()),
          reinterpret_cast<XPUType*>(logit_grad->data<T>()),
          use_softmax,
          n,
          d);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "soft_softmax_with_cross_entropy_grad");
    } else {
      xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
      int* labels_int_ptr_l3 =
          RAII_GUARD.alloc_l3_or_gm<int32_t>(labels.numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(labels_int_ptr_l3);

      r = xpu::cast_v2<int64_t, int32_t>(dev_ctx.x_context(),
                                         labels.data<int64_t>(),
                                         labels_int_ptr_l3,
                                         labels.numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast_v2");

      r = xpu::hard_softmax_with_cross_entropy_grad<XPUType, int>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(loss_grad.data<T>()),
          labels_int_ptr_l3,
          reinterpret_cast<const XPUType*>(softmax.data<T>()),
          reinterpret_cast<XPUType*>(logit_grad->data<T>()),
          ignore_index,
          use_softmax,
          n,
          d);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "hard_softmax_with_cross_entropy_grad");
    }
  } else {
    int t = logit_grad->dims()[axis];
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int len = softmax.numel();
    XPUType* trans_logit = RAII_GUARD.alloc_l3_or_gm<XPUType>(len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(trans_logit);

    XPUType* trans_softmax = RAII_GUARD.alloc_l3_or_gm<XPUType>(len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(trans_softmax);
    r = xpu::transpose(dev_ctx.x_context(),
                       reinterpret_cast<const XPUType*>(softmax.data<T>()),
                       trans_softmax,
                       {n, t, d / t},
                       {0, 2, 1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

    if (soft_label) {
      XPUType* trans_labels = RAII_GUARD.alloc_l3_or_gm<XPUType>(len);
      PADDLE_ENFORCE_XDNN_NOT_NULL(trans_labels);
      r = xpu::transpose(dev_ctx.x_context(),
                         reinterpret_cast<const XPUType*>(labels.data<T>()),
                         trans_labels,
                         {n, t, d / t},
                         {0, 2, 1});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
      r = xpu::soft_softmax_with_cross_entropy_grad<XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(loss_grad.data<T>()),
          trans_labels,
          trans_softmax,
          trans_logit,
          use_softmax,
          n * d / t,
          t);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "soft_softmax_with_cross_entropy_grad");
    } else {
      int* labels_int_ptr_l3 =
          RAII_GUARD.alloc_l3_or_gm<int32_t>(labels.numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(labels_int_ptr_l3);

      r = xpu::cast_v2<int64_t, int32_t>(dev_ctx.x_context(),
                                         labels.data<int64_t>(),
                                         labels_int_ptr_l3,
                                         labels.numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip_v2");
      r = xpu::hard_softmax_with_cross_entropy_grad<XPUType, int>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(loss_grad.data<T>()),
          labels_int_ptr_l3,
          trans_softmax,
          trans_logit,
          ignore_index,
          use_softmax,
          n * d / t,
          t);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "hard_softmax_with_cross_entropy_grad");
    }

    r = xpu::transpose<XPUType>(
        dev_ctx.x_context(),
        trans_logit,
        reinterpret_cast<XPUType*>(logit_grad->data<T>()),
        {n, d / t, t},
        {0, 2, 1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cross_entropy_with_softmax_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxGradKernel,
                   float,
                   phi::dtype::float16) {}
