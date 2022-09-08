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
  // check input
  PADDLE_ENFORCE_EQ(centered,
                    false,
                    errors::Unimplemented(
                        "centered=True is not supported in the xpu kernel of "
                        "rmsprop. use XPU_BLACK_LIST to disable this op."));
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

  // int rmsprop(Context* ctx, const T* g, const T* p, const float* ms, const
  // float* mom, T* p_out, float* ms_out, float* mom_out, float epsilon, float
  // rho, float momentum, float lr, int n);
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
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "rmsprop");
}
}  // namespace phi

PD_REGISTER_KERNEL(rmsprop, XPU, ALL_LAYOUT, phi::RmspropDenseKernel, float) {}
