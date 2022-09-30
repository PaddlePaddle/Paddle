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

#include "paddle/phi/kernels/transpose_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (out->numel() == 0) {
    return;
  }
  dev_ctx.template Alloc<T>(out);
  int ndims = axis.size();
  std::vector<int> x_shape_host(ndims, 0);
  for (int i = 0; i < ndims; ++i) {
    x_shape_host[i] = x.dims()[i];
  }
  int r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                  reinterpret_cast<const XPUType*>(x.data<T>()),
                                  reinterpret_cast<XPUType*>(out->data<T>()),
                                  x_shape_host,
                                  axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
}

}  // namespace phi

PD_REGISTER_KERNEL(transpose,
                   XPU,
                   ALL_LAYOUT,
                   phi::TransposeKernel,
                   float,
                   phi::dtype::float16) {}
