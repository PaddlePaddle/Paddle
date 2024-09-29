// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/logsumexp_grad_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void LogsumexpGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& dy,
                         const std::vector<int>& axis_in,
                         bool keepdim,
                         bool reduce_all,
                         DenseTensor* dx) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  reduce_all = recompute_reduce_all(x, axis_in, reduce_all);

  auto x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto y_data = reinterpret_cast<const XPUType*>(y.data<T>());
  auto dy_data = reinterpret_cast<const XPUType*>(dy.data<T>());
  auto dx_data = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(dx));

  std::vector<int64_t> xdims = common::vectorize<int64_t>(x.dims());
  std::vector<int64_t> ydims = xdims;

  if (reduce_all) {
    ydims = {1};
    xdims = {x.numel()};
  } else {
    std::vector<int64_t> axis;
    axis.reserve(axis_in.size());
    std::for_each(
        axis_in.begin(), axis_in.end(), [&axis, &xdims](const int& t) {
          if (t < 0) {
            axis.push_back(static_cast<int64_t>(t + xdims.size()));
          } else {
            axis.push_back(static_cast<int64_t>(t));
          }
        });
    for (size_t i = 0; i < axis.size(); ++i) {
      PADDLE_ENFORCE_LT(
          axis[i],
          ydims.size(),
          errors::InvalidArgument(
              "The axis should be less than the rank of Input(X)."));
      ydims[axis[i]] = 1;
    }
  }
  int64_t xlen = 1;
  for (size_t i = 0; i < xdims.size(); ++i) {
    PADDLE_ENFORCE_LT(0,
                      xdims[i],
                      errors::InvalidArgument(
                          "The dims of Input(X) should be greater than 0."));
    xlen *= xdims[i];
  }
  XPUType* tmp_data = RAII_GUARD.alloc<XPUType>(xlen);

  int ret = xpu::broadcast_sub(
      dev_ctx.x_context(), x_data, y_data, tmp_data, xdims, ydims);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast_sub");
  ret = xpu::exp(dev_ctx.x_context(), tmp_data, tmp_data, xlen);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "exp");
  ret = xpu::broadcast_mul(
      dev_ctx.x_context(), dy_data, tmp_data, dx_data, ydims, xdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast_mul");
}
}  // namespace phi

PD_REGISTER_KERNEL(logsumexp_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::LogsumexpGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
