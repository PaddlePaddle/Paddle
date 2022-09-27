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

#include "paddle/phi/kernels/mean_all_grad_kernel.h"

#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MeanAllGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& out_grad,
                       DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto OG = &out_grad;
  PADDLE_ENFORCE_EQ(
      OG->numel(),
      1,
      phi::errors::InvalidArgument("Mean Gradient should be scalar"));
  auto IG = x_grad;
  dev_ctx.template Alloc<T>(IG);

  XPUType* dx = reinterpret_cast<XPUType*>(IG->data<T>());

  const T* dy = OG->data<T>();
  T dy0_value;
  xpu_wait(dev_ctx.x_context()->xpu_stream);
  paddle::memory::Copy(phi::CPUPlace(), &dy0_value, OG->place(), dy, sizeof(T));
  float dy0_fp32 = static_cast<float>(dy0_value);
  dy0_fp32 = dy0_fp32 / static_cast<float>(IG->numel());

  int r = xpu::constant(
      dev_ctx.x_context(), dx, IG->numel(), static_cast<XPUType>(dy0_fp32));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mean_all_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(mean_all_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MeanAllGradKernel,
                   float,
                   phi::dtype::float16) {}
