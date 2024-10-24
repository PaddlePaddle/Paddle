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
  size_t x_rank = x.dims().size();
  std::vector<int> formatted_axis = axis;
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      formatted_axis[i] = axis[i] + x_rank;
    }
  }

  using XPUType = typename XPUTypeTrait<T>::Type;

  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  if (formatted_axis.size() == 0) {
    phi::Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  std::vector<int> x_dim_vec = common::vectorize<int>(x.dims());
  int r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                  reinterpret_cast<const XPUType*>(x.data<T>()),
                                  reinterpret_cast<XPUType*>(out->data<T>()),
                                  x_dim_vec,
                                  formatted_axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
}

}  // namespace phi

PD_REGISTER_KERNEL(transpose,
                   XPU,
                   ALL_LAYOUT,
                   phi::TransposeKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int,
                   bool) {}
