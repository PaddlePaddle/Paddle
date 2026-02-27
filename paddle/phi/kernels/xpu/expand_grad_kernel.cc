// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/expand_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T, typename Context>
void ExpandGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const IntArray& shape,
                      DenseTensor* in_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  if ((in_grad && in_grad->numel() == 0) || out_grad.numel() == 0) {
    Full<T, Context>(dev_ctx, in_grad->dims(), 0, in_grad);
    return;
  }

  auto in_grad_data = dev_ctx.template Alloc<T>(in_grad);
  auto out_grad_dims = vectorize<int64_t>(out_grad.dims());
  auto in_grad_dims = vectorize<int64_t>(in_grad->dims());
  in_grad_dims.insert(
      in_grad_dims.begin(), out_grad.dims().size() - in_grad->dims().size(), 1);

  // Two zero
  if (out_grad_dims.size() == 0 && in_grad_dims.size() == 0) {
    out_grad_dims = {1};
    in_grad_dims = {1};
  }

  int r = xpu::expand_grad<XPUType>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(out_grad.data<T>()),
      reinterpret_cast<XPUType*>(in_grad_data),
      out_grad_dims,
      in_grad_dims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "expand_grad");
}

template <>
void ExpandGradKernel<double, XPUContext>(const XPUContext& dev_ctx,
                                          const DenseTensor& x,
                                          const DenseTensor& out_grad,
                                          const IntArray& shape,
                                          DenseTensor* in_grad) {
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  if ((in_grad && in_grad->numel() == 0) || out_grad.numel() == 0) {
    Full<double, XPUContext>(dev_ctx, in_grad->dims(), 0, in_grad);
    return;
  }

  auto in_grad_data = dev_ctx.template Alloc<double>(in_grad);
  auto out_grad_dims = vectorize<int64_t>(out_grad.dims());
  auto in_grad_dims = vectorize<int64_t>(in_grad->dims());
  in_grad_dims.insert(
      in_grad_dims.begin(), out_grad.dims().size() - in_grad->dims().size(), 1);

  // Two zero
  if (out_grad_dims.size() == 0 && in_grad_dims.size() == 0) {
    out_grad_dims = {1};
    in_grad_dims = {1};
  }

  float* out_grad_fp32 = RAII_GUARD.alloc_l3_or_gm<float>(out_grad.numel());
  float* in_grad_fp32 = RAII_GUARD.alloc_l3_or_gm<float>(in_grad->numel());
  int r = 0;
  r = xpu::cast<double, float>(
      dev_ctx.x_context(),
      reinterpret_cast<const double*>(out_grad.data<double>()),
      out_grad_fp32,
      out_grad.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  r = xpu::expand_grad<float>(dev_ctx.x_context(),
                              out_grad_fp32,
                              in_grad_fp32,
                              out_grad_dims,
                              in_grad_dims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "expand_grad");
  r = xpu::cast<float, double>(dev_ctx.x_context(),
                               in_grad_fp32,
                               reinterpret_cast<double*>(in_grad_data),
                               in_grad->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
}
}  // namespace phi

PD_REGISTER_KERNEL(expand_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ExpandGradKernel,
                   float,
                   int64_t,
                   double,
                   phi::bfloat16,
                   phi::float16) {}
