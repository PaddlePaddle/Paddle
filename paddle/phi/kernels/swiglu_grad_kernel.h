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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename Context>
void SwiGLUGradKernelImpl(const Context &ctx,
                          const T *x,
                          const T *y,
                          const T *dz,
                          T *dx,
                          T *dy,
                          int64_t m,
                          int64_t n);

template <typename T, typename Context>
void SwiGLUGradKernel(const Context &ctx,
                      const DenseTensor &x,
                      const paddle::optional<DenseTensor> &y,
                      const DenseTensor &dz,
                      DenseTensor *dx,
                      DenseTensor *dy) {
  const auto *x_ptr = x.data<T>();
  const auto *dz_ptr = dz.data<T>();
  auto *dx_ptr = dx ? ctx.template Alloc<T>(dx) : nullptr;
  auto *dy_ptr = y && dy ? ctx.template Alloc<T>(dy) : nullptr;
  const auto &dims = x.dims();

  if (y) {
    const auto &y_tensor = y.get();
    const auto &y_dims = y_tensor.dims();
    PADDLE_ENFORCE_EQ(
        y_dims,
        dims,
        phi::errors::InvalidArgument("The shape of Input(Y):[%s] must be equal "
                                     "to the shape of Input(X):[%s].",
                                     y_dims,
                                     dims));
    SwiGLUGradKernelImpl<T, Context>(
        ctx, x_ptr, y_tensor.data<T>(), dz_ptr, dx_ptr, dy_ptr, x.numel(), 1);
  } else {
    auto dims_2d = flatten_to_2d(dims, dims.size() - 1);
    int64_t m = dims_2d[0], n = dims_2d[1];
    PADDLE_ENFORCE_EQ(n % 2,
                      0,
                      phi::errors::InvalidArgument(
                          "The last dim of Input(X) should be exactly divided "
                          "by 2 when Input(Y) is None, but got %d",
                          n));
    SwiGLUGradKernelImpl<T, Context>(
        ctx, x_ptr, nullptr, dz_ptr, dx_ptr, nullptr, m, n / 2);
  }
}

}  // namespace phi
