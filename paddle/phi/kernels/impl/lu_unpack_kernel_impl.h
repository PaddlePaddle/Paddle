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

#include "paddle/phi/kernels/funcs/lu.h"

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
  auto xin = ctx.Input<framework::Tensor>("X");
  auto P = ctx.Input<framework::Tensor>("Pivots");

  auto ltensor = ctx.Output<framework::Tensor>("L");
  auto utensor = ctx.Output<framework::Tensor>("U");
  auto ptensor = ctx.Output<framework::Tensor>("Pmat");

  auto unpack_ludata = ctx.Attr<bool>("unpack_ludata");
  auto unpack_pivots = ctx.Attr<bool>("unpack_pivots");

  const auto& dev_ctx = ctx.template device_context<DeviceContext>();

  auto xdims = xin->dims();
  int xrank = xdims.size();
  int64_t m = xdims[xrank - 2];
  int64_t n = xdims[xrank - 1];
  int64_t k = std::min(m, n);

  if (unpack_ludata) {
    ltensor->mutable_data<T>(ctx.GetPlace());
    utensor->mutable_data<T>(ctx.GetPlace());

    framework::Tensor L, U;
    LU_Unpack<DeviceContext, T>(dev_ctx, xin, &L, &U);

    if (m >= n) {
      framework::TensorCopy(L, ctx.GetPlace(), ltensor);
      Tensor_narrow<DeviceContext, T>(ctx, &U, utensor, 0, k, 0, k);
    } else {
      framework::TensorCopy(U, ctx.GetPlace(), utensor);
      Tensor_narrow<DeviceContext, T>(ctx, &L, ltensor, 0, k, 0, k);
    }
  }

  if (unpack_pivots) {
    ptensor->mutable_data<T>(ctx.GetPlace());
    Unpack_Pivot<DeviceContext, T>(dev_ctx, *P, ptensor, m, k);
  }
}

}  // namespace phi
