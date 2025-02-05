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

#include "paddle/phi/kernels/squared_l2_norm_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/common/memory_utils.h"

namespace phi {

template <typename T, typename Context>
void SquaredL2NormGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& dout,
                             DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  PADDLE_ENFORCE_EQ(
      dout.numel(),
      1,
      common::errors::InvalidArgument(
          "Input(GRAD@Out) of SquaredL2NormGradOP should be a scalar."));

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  using XPUType = typename XPUTypeTrait<T>::Type;
  XPUType dout_value_cpu = 0;
  memory_utils::Copy(CPUPlace(),
                     static_cast<void*>(&dout_value_cpu),
                     dev_ctx.GetPlace(),
                     static_cast<const void*>(dout.data<T>()),
                     sizeof(XPUType));

  // squared_l2_norm_grad: dx = dout(it is a scalar value!) * x * 2.0

  // int scale(Context* ctx, const T* x, T* y, int64_t len, bool
  // bias_after_scale, float _scale, float _bias);
  int r = xpu::scale(dev_ctx.x_context(),
                     reinterpret_cast<const XPUType*>(x.data<T>()),
                     reinterpret_cast<XPUType*>(dx->data<T>()),
                     x.numel(),
                     false,
                     dout_value_cpu * 2,
                     0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
}

}  // namespace phi

PD_REGISTER_KERNEL(squared_l2_norm_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SquaredL2NormGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
