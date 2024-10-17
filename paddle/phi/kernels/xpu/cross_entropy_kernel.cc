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
  const int rank = logits.dims().size();
  const int axis = phi::funcs::CanonicalAxis(axis_in, rank);
  dev_ctx.template Alloc<T>(softmax);
  dev_ctx.template Alloc<T>(loss);
  const int64_t n = phi::funcs::SizeToAxis(axis, logits.dims());
  const int64_t d = phi::funcs::SizeOutAxis(axis, logits.dims());
  const int64_t t = logits.dims()[axis];
  int64_t len = logits.numel();

  auto logits_data = reinterpret_cast<const XPUType*>(logits.data<T>());
  auto softmax_data = reinterpret_cast<XPUType*>(softmax->data<T>());
  auto loss_data = reinterpret_cast<XPUType*>(loss->data<T>());

  int r = XPU_SUCCESS;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  if (!use_softmax) {
    // For cross entropy only cases, logits are outputs of softmax
    // so we just copy input logits to the softmax output.
    r = xpu::copy<XPUType>(dev_ctx.x_context(), logits_data, softmax_data, len);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  } else if (d != 1) {
    // Because we transpose inputs when axis != logits.dims().size() - 1, we
    // need a temp buffer to save the transposed softmax.
    softmax_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(len);
  }
  if (d != 1) {
    // The XPU transpose API supports softmax with axis. However, we do the
    // transpose before softmax due to the following two reasons:
    // 1. the XPU cross_entropy APIs supports cross entropy on the last dim
    // only, so the transpose here is unavoidable for them.
    // 2. the XPU softmax api would do the transpose internally if axis is not
    // the last dim and we can eliminate a transpose call if we explicitly
    // transpose the inputs before the softmax calculation.
    XPUType* logits_trans = RAII_GUARD.alloc_l3_or_gm<XPUType>(len);
    r = xpu::transpose<XPUType>(
        dev_ctx.x_context(), logits_data, logits_trans, {n, t, d}, {0, 2, 1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    logits_data = logits_trans;
  }

  if (soft_label) {
    auto labels_data = reinterpret_cast<const XPUType*>(labels.data<T>());
    if (d != 1) {
      XPUType* labels_trans =
          RAII_GUARD.alloc_l3_or_gm<XPUType>(labels.numel());
      r = xpu::transpose<XPUType>(
          dev_ctx.x_context(), labels_data, labels_trans, {n, t, d}, {0, 2, 1});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    }
    if (use_softmax) {
      // 1. softmax + soft_cross_entropy
      r = xpu::soft_softmax_with_cross_entropy<XPUType>(dev_ctx.x_context(),
                                                        logits_data,
                                                        labels_data,
                                                        softmax_data,
                                                        loss_data,
                                                        n * d,
                                                        t);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "soft_softmax_with_cross_entropy");
    } else {
      r = xpu::soft_cross_entropy<XPUType>(
          dev_ctx.x_context(), logits_data, labels_data, loss_data, n * d, t);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "soft_cross_entropy");
    }
  } else {
    // 2. soft_cross_entropy only
    const int* labels_data = nullptr;
    if (labels.dtype() == phi::DataType::INT32) {
      labels_data = labels.data<int>();
    } else if (labels.dtype() == phi::DataType::INT64) {
      int* labels_tmp = RAII_GUARD.alloc_l3_or_gm<int>(labels.numel());
      r = xpu::cast<int64_t, int>(dev_ctx.x_context(),
                                  labels.data<int64_t>(),
                                  labels_tmp,
                                  labels.numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      labels_data = labels_tmp;
    } else {
      errors::Unimplemented(
          "Unsupported dtype for labels in hard cross entropy, only int32 and "
          "int64 are supported.");
    }
    if (use_softmax) {
      // 3. softmax+hard_cross_entropy
      // do not use the fusion api for performance reason now.
      r = xpu::softmax<XPUType>(
          dev_ctx.x_context(), logits_data, softmax_data, {n * d, t}, 1);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax");
    }
    // 4. hard_cross_entropy only
    r = xpu::hard_cross_entropy<XPUType, int>(dev_ctx.x_context(),
                                              softmax_data,
                                              labels_data,
                                              loss_data,
                                              nullptr,
                                              n * d,
                                              t,
                                              -100);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "hard_cross_entropy");
  }

  if (use_softmax && d != 1) {
    r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                softmax_data,
                                reinterpret_cast<XPUType*>(softmax->data<T>()),
                                {n, d, t},
                                {0, 2, 1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cross_entropy_with_softmax,
                   XPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
