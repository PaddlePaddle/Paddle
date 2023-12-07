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

#include "paddle/phi/kernels/where_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void WhereGradKernel(const Context& ctx,
                     const DenseTensor& condition,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out_grad,
                     DenseTensor* x_grad,
                     DenseTensor* y_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto* cond_data = condition.data<bool>();
  auto* dout = out_grad.data<T>();

  auto cond_shape = common::vectorize(condition.dims());
  auto out_shape = common::vectorize(out_grad.dims());
  // use [1] to replace [], because xpu not support []
  if (cond_shape.size() == 0) {
    cond_shape = std::vector<int64_t>({1});
  }
  if (out_shape.size() == 0) {
    out_shape = std::vector<int64_t>({1});
  }

  T* dx = nullptr;
  T* dy = nullptr;

  if (x_grad != nullptr) {
    dx = ctx.template Alloc<T>(x_grad);
  }

  if (y_grad != nullptr) {
    dy = ctx.template Alloc<T>(y_grad);
  }
  int r = xpu::select_grad(ctx.x_context(),
                           cond_data,
                           reinterpret_cast<const XPUType*>(dout),
                           reinterpret_cast<XPUType*>(dx),
                           reinterpret_cast<XPUType*>(dy),
                           cond_shape,
                           out_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "select_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(where_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::WhereGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
