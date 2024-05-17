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

#include "paddle/phi/kernels/transpose_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TransposeGradKernel(const Context& dev_ctx,
                         const DenseTensor& out_grad,
                         const std::vector<int>& axis,
                         DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(x_grad);
  if (x_grad->numel() == 0) {
    return;
  }

  size_t axis_size = axis.size();
  if (axis_size == 0) {
    phi::Copy<Context>(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    return;
  }

  std::vector<int> formatted_axis = axis;
  for (size_t i = 0; i < axis_size; i++) {
    if (axis[i] < 0) {
      formatted_axis[i] = axis[i] + axis_size;
    }
  }

  std::vector<int> reversed_axis(axis);
  for (size_t i = 0; i < axis_size; i++) {
    reversed_axis[formatted_axis[i]] = i;
  }

  std::vector<int> out_grad_dim_vec = common::vectorize<int>(out_grad.dims());
  int r = xpu::transpose<XPUType>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(out_grad.data<T>()),
      reinterpret_cast<XPUType*>(x_grad->data<T>()),
      out_grad_dim_vec,
      reversed_axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(transpose_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::TransposeGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int,
                   bool) {}
