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
void LUUnpackGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& pivots,
                        const DenseTensor& l,
                        const DenseTensor& u,
                        const DenseTensor& pmat,
                        const DenseTensor& l_grad,
                        const DenseTensor& u_grad,
                        bool unpack_ludata,
                        bool unpack_pivots,
                        DenseTensor* x_grad) {
  auto dl = ctx.Input<framework::Tensor>(framework::GradVarName("L"));
  auto du = ctx.Input<framework::Tensor>(framework::GradVarName("U"));
  auto dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
  dx->mutable_data<T>(ctx.GetPlace());

  const auto& dev_ctx = ctx.template device_context<DeviceContext>();

  framework::Tensor dl_tril, du_triu;
  const auto ldims = dl->dims();
  dl_tril.Resize(ldims);
  auto H = ldims[ldims.size() - 2];
  auto W = ldims[ldims.size() - 1];
  auto L_dataptr = dl_tril.mutable_data<T>(dev_ctx.GetPlace());
  platform::ForRange<DeviceContext> l_for_range(dev_ctx, dl->numel());
  phi::funcs::TrilTriuCompute<T> tril_computer(
      dl->data<T>(), -1, true, H, W, L_dataptr);
  l_for_range(tril_computer);

  const auto udims = du->dims();
  du_triu.Resize(udims);
  H = udims[udims.size() - 2];
  W = udims[udims.size() - 1];
  auto U_dataptr = du_triu.mutable_data<T>(dev_ctx.GetPlace());
  platform::ForRange<DeviceContext> u_for_range(dev_ctx, du->numel());
  phi::funcs::TrilTriuCompute<T> triu_computer(
      du->data<T>(), 0, false, H, W, U_dataptr);
  u_for_range(triu_computer);

  auto xdims = dx->dims();
  int xrank = xdims.size();
  int64_t m = xdims[xrank - 2];
  int64_t n = xdims[xrank - 1];
  int64_t k = std::min(m, n);

  std::vector<int64_t> axes = {xrank - 2, xrank - 1};
  std::vector<int64_t> slice_starts(2, 0);
  std::vector<int64_t> slice_ends(2, 0);
  auto valuedims = vectorize(xdims);

  phi::funcs::SetConstant<DeviceContext, T> setter;
  setter(dev_ctx, dx, static_cast<T>(0));
  if (m <= n) {
    slice_starts[0] = 0;
    slice_starts[1] = 0;
    slice_ends[0] = k;
    slice_ends[1] = k;
    valuedims[xrank - 2] = k;
    valuedims[xrank - 1] = k;
    SetValueCompute_dispatch<DeviceContext, T>(ctx,
                                               dx,
                                               &dl_tril,
                                               dx,
                                               axes,
                                               &slice_starts,
                                               &slice_ends,
                                               valuedims,
                                               xrank);

    Tensor_Add<DeviceContext, T>(dev_ctx, *dx, du_triu, dx);
  } else {
    slice_starts[0] = 0;
    slice_starts[1] = 0;
    slice_ends[0] = k;
    slice_ends[1] = k;
    valuedims[xrank - 2] = k;
    valuedims[xrank - 1] = k;
    SetValueCompute_dispatch<DeviceContext, T>(ctx,
                                               dx,
                                               &du_triu,
                                               dx,
                                               axes,
                                               &slice_starts,
                                               &slice_ends,
                                               valuedims,
                                               xrank);

    Tensor_Add<DeviceContext, T>(dev_ctx, *dx, dl_tril, dx);
  }
}

}  // namespace phi
