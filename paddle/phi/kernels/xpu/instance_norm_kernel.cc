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

#include "paddle/phi/kernels/instance_norm_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void InstanceNormKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const paddle::optional<DenseTensor>& scale,
                        const paddle::optional<DenseTensor>& bias,
                        float epsilon,
                        DenseTensor* y,
                        DenseTensor* saved_mean,
                        DenseTensor* saved_var) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const auto& x_dims = x.dims();
  int n = x_dims[0];
  int c = x_dims[1];
  int h = x_dims[2];
  int w = x_dims[3];
  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<float>(saved_mean);
  dev_ctx.template Alloc<float>(saved_var);

  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();

  int r = xpu::instance_norm(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(x.data<T>()),
                             reinterpret_cast<XPUType*>(y->data<T>()),
                             n,
                             c,
                             h,
                             w,
                             epsilon,
                             scale_ptr->data<float>(),
                             bias_ptr->data<float>(),
                             saved_mean->data<float>(),
                             saved_var->data<float>(),
                             true);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "instance_norm");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    instance_norm, XPU, ALL_LAYOUT, phi::InstanceNormKernel, float) {}
