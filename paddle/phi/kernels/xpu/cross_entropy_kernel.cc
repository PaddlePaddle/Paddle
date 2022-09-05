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

#include "paddle/phi/kernels/cross_entropy_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace phi {

template <typename T, typename Context>
void CrossEntropyWithSoftmaxKernel(const Context& dev_ctx,
                                   const DenseTensor& logits,
                                   const DenseTensor& labels,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis_in,
                                   DenseTensor* softmax,
                                   DenseTensor* loss) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  PADDLE_ENFORCE_EQ(
      logits.place().GetType() == phi::AllocationType::XPU,
      true,
      errors::PreconditionNotMet("This kernel only runs on XPU."));

  const int rank = logits.dims().size();
  const int axis = phi::funcs::CanonicalAxis(axis_in, rank);
  dev_ctx.template Alloc<T>(softmax);
  dev_ctx.template Alloc<T>(loss);
  const int n = phi::funcs::SizeToAxis(axis, logits.dims());
  const int d = phi::funcs::SizeFromAxis(axis, logits.dims());
  std::vector<int> logits_dims = phi::vectorize<int>(logits.dims());

  int t = logits_dims[axis];

  auto logits_data = reinterpret_cast<const XPUType*>(logits.data<T>());
  auto softmax_data = reinterpret_cast<XPUType*>(softmax->data<T>());
  auto loss_data = reinterpret_cast<XPUType*>(loss->data<T>());
  // softmax
  int r = XPU_SUCCESS;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  if (phi::backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId()) ==
          phi::backends::xpu::XPUVersion::XPU2 &&
      soft_label && axis == rank - 1) {
    auto labels_data = reinterpret_cast<const XPUType*>(labels.data<T>());
    r = xpu::soft_softmax_with_cross_entropy<XPUType>(dev_ctx.x_context(),
                                                      logits_data,
                                                      labels_data,
                                                      softmax_data,
                                                      loss_data,
                                                      n,
                                                      d);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "soft_softmax_with_cross_entropy");
    return;
  }

  int len = logits.numel();
  T* clip_logits = RAII_GUARD.alloc_l3_or_gm<T>(len);
  PADDLE_ENFORCE_XDNN_NOT_NULL(clip_logits);
  XPUType* clip_logits_data = reinterpret_cast<XPUType*>(clip_logits);

  float max_val = 1e20;
  float min_val = -1e20;
  if (std::is_same<T, dtype::float16>::value) {
    max_val = 65504;
    min_val = -65504;
  }

  r = xpu::clip_v2<XPUType>(dev_ctx.x_context(),
                            logits_data,
                            clip_logits_data,
                            len,
                            static_cast<XPUType>(min_val),
                            static_cast<XPUType>(max_val));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip_v2");

  r = xpu::softmax<XPUType>(
      dev_ctx.x_context(), clip_logits_data, softmax_data, logits_dims, axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax");

  // cross_entropy
  if (axis != rank - 1) {
    XPUType* trans_softmax = RAII_GUARD.alloc_l3_or_gm<XPUType>(n * d);
    PADDLE_ENFORCE_XDNN_NOT_NULL(trans_softmax);

    r = xpu::transpose(dev_ctx.x_context(),
                       softmax_data,
                       trans_softmax,
                       {n, t, d / t},
                       {0, 2, 1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    softmax_data = trans_softmax;
  }

  if (soft_label) {
    auto labels_data = reinterpret_cast<const XPUType*>(labels.data<T>());
    if (axis != rank - 1) {
      XPUType* trans_label = RAII_GUARD.alloc_l3_or_gm<XPUType>(n * d);
      PADDLE_ENFORCE_XDNN_NOT_NULL(trans_label);
      r = xpu::transpose(dev_ctx.x_context(),
                         labels_data,
                         trans_label,
                         {n, t, d / t},
                         {0, 2, 1});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
      labels_data = trans_label;
    }
    r = xpu::soft_cross_entropy<XPUType>(dev_ctx.x_context(),
                                         softmax_data,
                                         labels_data,
                                         loss_data,
                                         axis == rank - 1 ? n : n * d / t,
                                         axis == rank - 1 ? d : t);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "soft_cross_entropy");
  } else {
    DenseTensor labels_int32;
    int* labels_int_ptr_l3 = RAII_GUARD.alloc_l3_or_gm<int32_t>(labels.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(labels_int_ptr_l3);

    r = xpu::cast_v2<int64_t, int32_t>(dev_ctx.x_context(),
                                       labels.data<int64_t>(),
                                       labels_int_ptr_l3,
                                       labels.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip_v2");

    r = xpu::hard_cross_entropy<XPUType, int32_t>(
        dev_ctx.x_context(),
        softmax_data,
        labels_int_ptr_l3,
        loss_data,
        nullptr,
        axis == rank - 1 ? n : n * d / t,
        axis == rank - 1 ? d : t,
        ignore_index);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "hard_cross_entropy");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cross_entropy_with_softmax,
                   XPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxKernel,
                   float,
                   phi::dtype::float16) {}
