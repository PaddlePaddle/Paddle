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

#include "paddle/phi/kernels/where_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void WhereKernel(const Context& ctx,
                 const DenseTensor& condition,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const bool* cond_data = condition.data<bool>();
  const XPUType* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* y_data = reinterpret_cast<const XPUType*>(y.data<T>());
  XPUType* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));

  auto cond_dims = common::vectorize<int>(condition.dims());
  auto x_dims = common::vectorize<int>(x.dims());

  // use [1] to replace [], because xpu not support []
  if (cond_dims.size() == 0) {
    cond_dims = std::vector<int>({1});
  }
  if (x_dims.size() == 0) {
    x_dims = std::vector<int>({1});
  }

  int ret = xpu::select(
      ctx.x_context(), cond_data, x_data, y_data, out_data, cond_dims, x_dims);

  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "xpu::select");
}

}  // namespace phi

PD_REGISTER_KERNEL(where,
                   XPU,
                   ALL_LAYOUT,
                   phi::WhereKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
