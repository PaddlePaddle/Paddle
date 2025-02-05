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

#pragma once

#include "paddle/phi/kernels/impl/lu_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void LUUnpackKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& pivots,
                    bool unpack_ludata,
                    bool unpack_pivots,
                    DenseTensor* pmat,
                    DenseTensor* l,
                    DenseTensor* u) {
  auto xdims = x.dims();
  int xrank = xdims.size();
  int64_t m = xdims[xrank - 2];
  int64_t n = xdims[xrank - 1];
  int64_t k = std::min(m, n);

  if (unpack_ludata) {
    dev_ctx.template Alloc<T>(l);
    dev_ctx.template Alloc<T>(u);

    DenseTensor L, U;
    LU_Unpack<Context, T>(dev_ctx, &x, &L, &U);

    if (m >= n) {
      phi::Copy(dev_ctx, L, dev_ctx.GetPlace(), false, l);
      Tensor_narrow<Context, T>(dev_ctx, &U, u, 0, k, 0, k);
    } else {
      phi::Copy(dev_ctx, U, dev_ctx.GetPlace(), false, u);
      Tensor_narrow<Context, T>(dev_ctx, &L, l, 0, k, 0, k);
    }
  }

  if (unpack_pivots) {
    dev_ctx.template Alloc<T>(pmat);

    PADDLE_ENFORCE_EQ(
        pivots.dtype(),
        phi::DataType::INT32,
        common::errors::InvalidArgument(
            "The pivots of lu_unpack must be of type int32, but received [%s].",
            pivots.dtype()));

    Unpack_Pivot<Context, T>(dev_ctx, pivots, pmat, m, k);
  }
}

}  // namespace phi
