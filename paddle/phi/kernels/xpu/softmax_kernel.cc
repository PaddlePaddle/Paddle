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

#include "paddle/phi/kernels/softmax_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace phi {

template <typename T, typename Context>
void SoftmaxKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   int axis,
                   DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const int rank = x.dims().size();
  const int calc_axis = phi::funcs::CanonicalAxis(axis, rank);

  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  std::vector<int> x_dims;
  for (int i = 0; i < rank; i++) {
    x_dims.push_back(x.dims()[i]);
  }

  int r = XPU_SUCCESS;
  auto version =
      phi::backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId());
  if (version == phi::backends::xpu::XPUVersion::XPU1) {
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    XPUType* clip_x_data_l3 = RAII_GUARD.alloc_l3_or_gm<XPUType>(x.numel());
    r = xpu::clip_v2(dev_ctx.x_context(),
                     reinterpret_cast<const XPUType*>(x.data<T>()),
                     clip_x_data_l3,
                     x.numel(),
                     static_cast<XPUType>(-1e20),
                     static_cast<XPUType>(1e20));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip_v2");
    r = xpu::softmax<XPUType>(dev_ctx.x_context(),
                              clip_x_data_l3,
                              reinterpret_cast<XPUType*>(out->data<T>()),
                              x_dims,
                              calc_axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax");
  } else {
    r = xpu::softmax<XPUType>(dev_ctx.x_context(),
                              reinterpret_cast<const XPUType*>(x.data<T>()),
                              reinterpret_cast<XPUType*>(out->data<T>()),
                              x_dims,
                              calc_axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    softmax, XPU, ALL_LAYOUT, phi::SoftmaxKernel, float, phi::dtype::float16) {}
