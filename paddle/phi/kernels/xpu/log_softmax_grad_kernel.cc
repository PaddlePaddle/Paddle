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

namespace phi {

template <typename T, typename Context>
void LogSoftmaxGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& out_grad,
                          int axis,
                          DenseTensor* x_grad) {
  const int rank = out.dims().size();
  axis = funcs::CanonicalAxis(axis, rank);

  if (out.numel() != 0) {
    auto out_shape = phi::vectorize<int>(out.dims());
    x_grad->mutable_data<float>(dev_ctx.GetPlace());
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    T* tmp_ptr = RAII_GUARD.alloc_l3_or_gm<T>(out_grad.numel());
    T* tmp2_ptr = RAII_GUARD.alloc_l3_or_gm<T>(out_grad.numel());

    int r =
        xpu::exp(dev_ctx.x_context(), out.data<T>(), tmp_ptr, out_grad.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "exp");
    r = xpu::reciprocal(
        dev_ctx.x_context(), tmp_ptr, tmp2_ptr, out_grad.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reciprocal");
    r = xpu::mul(dev_ctx.x_context(),
                 tmp2_ptr,
                 out_grad.data<T>(),
                 tmp2_ptr,
                 out_grad.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reciprocal");
    r = xpu::softmax_grad(dev_ctx.x_context(),
                          tmp_ptr,
                          tmp2_ptr,
                          x_grad->data<T>(),
                          out_shape,
                          axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    log_softmax_grad, XPU, ALL_LAYOUT, phi::LogSoftmaxGradKernel, float) {}
