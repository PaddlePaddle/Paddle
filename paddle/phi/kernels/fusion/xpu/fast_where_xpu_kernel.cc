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

#include "glog/logging.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FastWhereXPUKernel(const Context& ctx,
                        const DenseTensor& condition,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* condition_data = condition.data<bool>();
  auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto* y_data = reinterpret_cast<const XPUType*>(y.data<T>());
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  auto condition_dims = common::vectorize<int>(condition.dims());
  auto x_dims = common::vectorize<int>(x.dims());
  auto y_dims = common::vectorize<int>(y.dims());
  PADDLE_ENFORCE_EQ(
      x_dims,
      y_dims,
      errors::PreconditionNotMet(
          "The dimensions of inputs should be equal, but x_dims=[",
          x.dims(),
          "] and y_dims=[",
          y.dims(),
          "]"));
#ifndef PADDLE_WITH_XPU_PLUGIN
  LOG(INFO)
      << "Add -DWITH_XPU_PLUGIN=ON to build xpu::plugin::fast_where(), or use "
         "xpu::select() instead, which leads low performance.";
  int r = xpu::select<XPUType>(ctx.x_context(),
                               condition_data,
                               x_data,
                               y_data,
                               out_data,
                               condition_dims,
                               x_dims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "select");
#else
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  if (condition_dims != x_dims) {
    bool* temp_data = RAII_GUARD.alloc_l3_or_gm<bool>(x.numel());
    int r = xpu::broadcast<bool>(
        ctx.x_context(), condition_data, temp_data, condition_dims, x_dims);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
    condition_data = temp_data;
  }
  int r = xpu::plugin::fast_where<XPUType>(
      ctx.x_context(), condition_data, x_data, y_data, out_data, x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "fast_where");
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fast_where_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FastWhereXPUKernel,
                   float,
                   phi::dtype::float16,
                   int) {}
