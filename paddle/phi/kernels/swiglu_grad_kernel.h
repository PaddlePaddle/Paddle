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
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T, typename Context>
void SwiGLUGradKernelImpl(const Context &dev_ctx,
                          const T *x,
                          const T *y,
                          const T *dz,
                          T *dx,
                          T *dy,
                          int64_t m,
                          int64_t n);

template <typename T, typename Context>
void SwiGLUGradKernel(const Context &dev_ctx,
                      const DenseTensor &x,
                      const optional<DenseTensor> &y,
                      const DenseTensor &dz,
                      DenseTensor *dx,
                      DenseTensor *dy) {
  const auto &dims = x.dims();
  const auto &dz_dims = dz.dims();
  // Validate shapes before the 0-size early returns below, otherwise a
  // mismatched dz/y is silently accepted whenever x has no elements.
  int64_t m = 0, n = 0;
  if (y) {
    const auto &y_dims = y.get().dims();
    // Mirror SwiGLUInferMeta: x and y may differ when either is 0-size, and
    // out (hence dz) then takes y's shape only if x is the non-empty one.
    const bool x_empty = x.numel() == 0;
    const bool y_empty = y.get().numel() == 0;
    if (!x_empty && !y_empty) {
      PADDLE_ENFORCE_EQ(y_dims,
                        dims,
                        common::errors::InvalidArgument(
                            "The shape of Input(Y):[%s] must be equal "
                            "to the shape of Input(X):[%s].",
                            y_dims,
                            dims));
    }
    const auto &out_dims = (!x_empty && y_empty) ? y_dims : dims;
    PADDLE_ENFORCE_EQ(dz_dims,
                      out_dims,
                      common::errors::InvalidArgument(
                          "The shape of Input(dz):[%s] must be equal "
                          "to the shape of Out:[%s].",
                          dz_dims,
                          out_dims));
  } else {
    auto dims_2d = flatten_to_2d(dims, dims.size() - 1);
    m = dims_2d[0];
    n = dims_2d[1];
    PADDLE_ENFORCE_EQ(n % 2,
                      0,
                      common::errors::InvalidArgument(
                          "The last dim of Input(X) should be exactly divided "
                          "by 2 when Input(Y) is None, but got %d",
                          n));
    PADDLE_ENFORCE_EQ(
        dz_dims.size(),
        dims.size(),
        common::errors::InvalidArgument(
            "The rank of Input(dz):[%d] must be equal to the rank of "
            "Input(X):[%d] when Input(Y) is None.",
            dz_dims.size(),
            dims.size()));
    for (int i = 0; i < dims.size() - 1; ++i) {
      PADDLE_ENFORCE_EQ(dz_dims[i],
                        dims[i],
                        common::errors::InvalidArgument(
                            "The shape of Input(dz):[%s] must be equal to "
                            "the shape of Input(X):[%s] except the last dim "
                            "when Input(Y) is None.",
                            dz_dims,
                            dims));
    }
    PADDLE_ENFORCE_EQ(
        dz_dims[dz_dims.size() - 1],
        n / 2,
        common::errors::InvalidArgument(
            "The last dim of Input(dz):[%d] must be equal to half of the "
            "last dim of Input(X):[%d] when Input(Y) is None.",
            dz_dims[dz_dims.size() - 1],
            n));
  }

  if (dx && dx->numel() == 0) {
    dev_ctx.template Alloc<T>(dx);
    if (dy) {
      Full<T, Context>(dev_ctx, dy->dims(), 0, dy);
    }
    return;
  }

  if (dy && dy->numel() == 0) {
    dev_ctx.template Alloc<T>(dy);
    if (dx) {
      Full<T, Context>(dev_ctx, dx->dims(), 0, dx);
    }
    return;
  }

  const auto *x_ptr = x.data<T>();
  const auto *dz_ptr = dz.data<T>();
  auto *dx_ptr = dx ? dev_ctx.template Alloc<T>(dx) : nullptr;
  auto *dy_ptr = y && dy ? dev_ctx.template Alloc<T>(dy) : nullptr;

  if (y) {
    SwiGLUGradKernelImpl<T, Context>(dev_ctx,
                                     x_ptr,
                                     y.get().data<T>(),
                                     dz_ptr,
                                     dx_ptr,
                                     dy_ptr,
                                     x.numel(),
                                     1);
  } else {
    SwiGLUGradKernelImpl<T, Context>(
        dev_ctx, x_ptr, nullptr, dz_ptr, dx_ptr, nullptr, m, n / 2);
  }
}

}  // namespace phi
