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

#include "paddle/phi/kernels/adadelta_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void AdadeltaKernel(const Context& dev_ctx,
                    const DenseTensor& param,
                    const DenseTensor& grad,
                    const DenseTensor& avg_squared_grad,
                    const DenseTensor& avg_squared_update,
                    float rho,
                    float epsilon,
                    DenseTensor* param_out,
                    DenseTensor* avg_squared_grad_out,
                    DenseTensor* avg_squared_update_out) {
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(avg_squared_grad_out);
  dev_ctx.template Alloc<T>(avg_squared_update_out);

  int r = xpu::adadelta<T, T>(dev_ctx.x_context(),
                              param.data<T>(),
                              grad.data<T>(),
                              avg_squared_grad.data<T>(),
                              avg_squared_update.data<T>(),
                              param_out->data<T>(),
                              avg_squared_grad_out->data<T>(),
                              avg_squared_update_out->data<T>(),
                              param.numel(),
                              rho,
                              epsilon);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "adadelta");
}

}  // namespace phi

PD_REGISTER_KERNEL(adadelta, XPU, ALL_LAYOUT, phi::AdadeltaKernel, float) {}
