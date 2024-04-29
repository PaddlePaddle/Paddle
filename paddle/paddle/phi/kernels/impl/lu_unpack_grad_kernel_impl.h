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
void LUUnpackGradKernel(const Context& dev_ctx,
                        const DenseTensor& x UNUSED,
                        const DenseTensor& pivots UNUSED,
                        const DenseTensor& l UNUSED,
                        const DenseTensor& u UNUSED,
                        const DenseTensor& pmat UNUSED,
                        const DenseTensor& l_grad,
                        const DenseTensor& u_grad,
                        bool unpack_ludata UNUSED,
                        bool unpack_pivots UNUSED,
                        DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  DenseTensor dl_tril, du_triu;
  const auto ldims = l_grad.dims();
  dl_tril.Resize(ldims);
  auto H = ldims[ldims.size() - 2];
  auto W = ldims[ldims.size() - 1];
  dev_ctx.template Alloc<T>(&dl_tril);
  auto L_dataptr = dl_tril.data<T>();
  phi::funcs::ForRange<Context> l_for_range(dev_ctx, l_grad.numel());
  phi::funcs::TrilTriuCompute<T> tril_computer(
      l_grad.data<T>(), -1, true, H, W, L_dataptr);
  l_for_range(tril_computer);

  const auto udims = u_grad.dims();
  du_triu.Resize(udims);
  H = udims[udims.size() - 2];
  W = udims[udims.size() - 1];
  dev_ctx.template Alloc<T>(&du_triu);
  auto U_dataptr = du_triu.data<T>();
  phi::funcs::ForRange<Context> u_for_range(dev_ctx, u_grad.numel());
  phi::funcs::TrilTriuCompute<T> triu_computer(
      u_grad.data<T>(), 0, false, H, W, U_dataptr);
  u_for_range(triu_computer);

  auto xdims = x_grad->dims();
  int xrank = xdims.size();
  int64_t m = xdims[xrank - 2];
  int64_t n = xdims[xrank - 1];
  int64_t k = std::min(m, n);

  std::vector<int64_t> axes = {xrank - 2, xrank - 1};
  std::vector<int64_t> slice_starts(2, 0);
  std::vector<int64_t> slice_ends(2, 0);
  auto valuedims = common::vectorize(xdims);

  phi::funcs::SetConstant<Context, T> setter;
  setter(dev_ctx, x_grad, static_cast<T>(0));
  if (m <= n) {
    slice_starts[0] = 0;
    slice_starts[1] = 0;
    slice_ends[0] = k;
    slice_ends[1] = k;
    valuedims[xrank - 2] = k;
    valuedims[xrank - 1] = k;
    SetValueCompute_dispatch<Context, T>(dev_ctx,
                                         x_grad,
                                         &dl_tril,
                                         x_grad,
                                         axes,
                                         &slice_starts,
                                         &slice_ends,
                                         valuedims,
                                         xrank);

    Tensor_Add<Context, T>(dev_ctx, *x_grad, du_triu, x_grad);
  } else {
    slice_starts[0] = 0;
    slice_starts[1] = 0;
    slice_ends[0] = k;
    slice_ends[1] = k;
    valuedims[xrank - 2] = k;
    valuedims[xrank - 1] = k;
    SetValueCompute_dispatch<Context, T>(dev_ctx,
                                         x_grad,
                                         &du_triu,
                                         x_grad,
                                         axes,
                                         &slice_starts,
                                         &slice_ends,
                                         valuedims,
                                         xrank);

    Tensor_Add<Context, T>(dev_ctx, *x_grad, dl_tril, x_grad);
  }
}

}  // namespace phi
