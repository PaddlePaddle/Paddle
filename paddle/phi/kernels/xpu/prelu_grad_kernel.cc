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

#include "paddle/phi/kernels/prelu_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void PReluGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& alpha,
                     const DenseTensor& out_grad,
                     const std::string& data_format,
                     const std::string& mode,
                     DenseTensor* x_grad,
                     DenseTensor* alpha_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const T* x_ptr = x.data<T>();
  const T* alpha_ptr = alpha.data<T>();
  const T* out_grad_ptr = out_grad.data<T>();

  T* x_grad_ptr = dev_ctx.template Alloc<T>(x_grad);
  T* alpha_grad_ptr = dev_ctx.template Alloc<T>(alpha_grad);

  auto x_dim = x.dims();
  auto x_rank = x_dim.size();

  std::vector<int> x_shape(x_rank);
  if (x_rank == 0) {
    x_shape = std::vector<int>({1});
  } else {
    for (int i = 0; i < x_rank; i++) {
      x_shape[i] = x_dim[i];
    }
  }

  auto alpha_dim = alpha.dims();
  auto alpha_rank = alpha_dim.size();

  std::vector<int> alpha_shape(alpha_rank);
  if (alpha_rank == 0) {
    alpha_shape = std::vector<int>({1});
  } else {
    for (int i = 0; i < x_rank; i++) {
      alpha_shape[i] = alpha_dim[i];
    }
  }

  // mode = 0: channel_nchw, slope_shape = {c}, default. meanwhile, xshape = {n,
  // c, h, w}
  // mode = 1, channel_nhwc, slope_shape = {c}, meanwhile, xshape = {n, h, w, c}
  // mode = 2, elementwise, slope_shape = {c*h*w}
  // mode = 3, single slope, slope_shape = {1}

  int xpu_mode = 0;

  if (mode == "channel") {
    if (data_format == "NCHW") {
      xpu_mode = 0;
    } else {
      // NHWC
      xpu_mode = 1;
    }
  } else if (mode == "element") {
    xpu_mode = 2;
  } else {
    xpu_mode = 3;
  }

  int r = xpu::prelu_grad(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x_ptr),
      reinterpret_cast<const XPUType*>(
          out_grad_ptr), /* const T* y, not used in xpu kernel */
      reinterpret_cast<const XPUType*>(alpha_ptr),
      reinterpret_cast<const XPUType*>(out_grad_ptr),
      reinterpret_cast<XPUType*>(x_grad_ptr),
      reinterpret_cast<XPUType*>(alpha_grad_ptr),
      x_shape,
      xpu_mode);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "prelu_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(prelu_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::PReluGradKernel,
                   float,
                   phi::dtype::float16) {}
