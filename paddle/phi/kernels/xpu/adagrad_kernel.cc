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

#include "paddle/phi/kernels/adagrad_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void AdagradDenseKernel(const Context& dev_ctx,
                        const DenseTensor& param,
                        const DenseTensor& grad,
                        const DenseTensor& moment,
                        const DenseTensor& learning_rate,
                        const paddle::optional<DenseTensor>& master_param,
                        float epsilon_t,
                        bool multi_precision,
                        DenseTensor* param_out_tensor,
                        DenseTensor* moment_out_tensor,
                        DenseTensor* master_param_outs) {
  dev_ctx.template Alloc<T>(param_out_tensor);
  dev_ctx.template Alloc<T>(moment_out_tensor);

  T epsilon = static_cast<T>(epsilon_t);

  int r = xpu::adagrad(dev_ctx.x_context(),
                       param.data<T>(),
                       grad.data<T>(),
                       moment.data<T>(),
                       learning_rate.data<T>(),
                       param_out_tensor->data<T>(),
                       moment_out_tensor->data<T>(),
                       param.numel(),
                       epsilon);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "adagrad");
}

}  // namespace phi

PD_REGISTER_KERNEL(adagrad, XPU, ALL_LAYOUT, phi::AdagradDenseKernel, float) {}
