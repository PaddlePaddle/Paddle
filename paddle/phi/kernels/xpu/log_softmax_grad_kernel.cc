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

#include "paddle/phi/kernels/log_softmax_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void LogSoftmaxGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& out_grad,
                          int axis,
                          DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const int rank = out.dims().size();
  axis = funcs::CanonicalAxis(axis, rank);

  // For 0D Tensor
  if (rank == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    phi::funcs::set_constant(dev_ctx, x_grad, static_cast<T>(0.0));
    return;
  }

  auto out_shape = common::vectorize<int>(out.dims());
  dev_ctx.template Alloc<T>(x_grad);
  int r = xpu::log_softmax_grad(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(out.data<T>()),
      reinterpret_cast<const XPUType*>(out_grad.data<T>()),
      reinterpret_cast<XPUType*>(x_grad->data<T>()),
      out_shape,
      axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "log_softmax_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    log_softmax_grad, XPU, ALL_LAYOUT, phi::LogSoftmaxGradKernel, float) {}
