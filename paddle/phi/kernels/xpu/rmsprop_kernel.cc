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

#include "paddle/phi/kernels/rmsprop_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/memory/memcpy.h"

namespace phi {

template <typename T, typename Context>
void RmspropDenseKernel(const Context& dev_ctx,
                        const DenseTensor& param,
                        const DenseTensor& mean_square,
                        const DenseTensor& grad,
                        const DenseTensor& moment,
                        const DenseTensor& learning_rate,
                        const paddle::optional<DenseTensor>& mean_grad,
                        float epsilon,
                        float decay,
                        float momentum,
                        bool centered,
                        DenseTensor* param_out,
                        DenseTensor* moment_out,
                        DenseTensor* mean_square_out,
                        DenseTensor* mean_grad_out) {
  // copy learning_rate to cpu
  PADDLE_ENFORCE_EQ(
      learning_rate.dims().size(),
      1,
      errors::InvalidArgument("learining rate should have dimension = 1."
                              " But received learning rate dim [%s] ",
                              learning_rate.dims().size()));
  T learning_rate_cpu = 0.0f;
  paddle::memory::Copy(CPUPlace(),
                       static_cast<void*>(&learning_rate_cpu),
                       dev_ctx.GetPlace(),
                       static_cast<const void*>(learning_rate.data()),
                       sizeof(T));

  // alloc output
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(moment_out);
  dev_ctx.template Alloc<T>(mean_square_out);

  if (centered) {
    dev_ctx.template Alloc<T>(mean_grad_out);
    auto mg_tensor = mean_grad.get_ptr();
    if (mg_tensor) {
      PADDLE_ENFORCE_EQ(
          mg_tensor->Holder(),
          mean_grad_out->Holder(),
          phi::errors::InvalidArgument(
              "MeanGrad and MeanGradOut must be the same Tensor"));
    } else {
      PADDLE_ENFORCE_EQ(
          mg_tensor,
          mean_grad_out,
          phi::errors::InvalidArgument(
              "MeanGrad and MeanGradOut must be the same Tensor"));
    }
    int r = xpu::rmsprop(dev_ctx.x_context(),
                         grad.data<T>(),
                         param.data<T>(),
                         mean_square.data<T>(),
                         moment.data<T>(),
                         param_out->data<T>(),
                         mean_square_out->data<T>(),
                         moment_out->data<T>(),
                         epsilon,
                         decay,
                         momentum,
                         learning_rate_cpu,
                         param.numel(),
                         centered,
                         mg_tensor->data<T>(),
                         mean_grad_out->data<T>());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "centered rmsprop");

  } else {
    int r = xpu::rmsprop(dev_ctx.x_context(),
                         grad.data<T>(),
                         param.data<T>(),
                         mean_square.data<T>(),
                         moment.data<T>(),
                         param_out->data<T>(),
                         mean_square_out->data<T>(),
                         moment_out->data<T>(),
                         epsilon,
                         decay,
                         momentum,
                         learning_rate_cpu,
                         param.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "uncentered rmsprop");
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(rmsprop, XPU, ALL_LAYOUT, phi::RmspropDenseKernel, float) {}
