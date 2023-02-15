/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/softmax_grad_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace phi {

template <typename T, typename Context>
void SoftmaxGradKernel(const Context& dev_ctx,
                       const DenseTensor& out,
                       const DenseTensor& out_grad,
                       int axis,
                       DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const int rank = x_grad->dims().size();
  const int calc_axis = phi::funcs::CanonicalAxis(axis, rank);

  // allocate memory on device.
  dev_ctx.template Alloc<T>(x_grad);
  if (x_grad->numel() == 0) {
    return;
  }

  std::vector<int> x_dims;
  for (int i = 0; i < rank; i++) {
    x_dims.push_back(x_grad->dims()[i]);
  }

  int r = xpu::softmax_grad<XPUType>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(out.data<T>()),
      reinterpret_cast<const XPUType*>(out_grad.data<T>()),
      reinterpret_cast<XPUType*>(x_grad->data<T>()),
      x_dims,
      calc_axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(softmax_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SoftmaxGradKernel,
                   float,
                   phi::dtype::float16) {}
