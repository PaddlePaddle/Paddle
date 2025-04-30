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

#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/frobenius_norm_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"

namespace phi {

template <typename T, typename Context>
void FrobeniusNormKernel(const Context& ctx,
                         const DenseTensor& x,
                         const IntArray& axis,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* out) {
  auto xdim = x.dims();

  if (x.numel() == 0) {
    std::set<int> axis_set;
    for (int ax : axis.GetData()) {
      if (ax < 0) {
        ax += xdim.size();
      }
      axis_set.insert(ax);
    }

    std::vector<int64_t> out_dims_vec;
    for (int i = 0; i < xdim.size(); ++i) {
      if (axis_set.find(i) == axis_set.end()) {
        out_dims_vec.push_back(xdim[i]);
      } else if (keep_dim) {
        out_dims_vec.push_back(1);
      }
    }
    out->Resize(phi::make_ddim(out_dims_vec));
    ctx.template Alloc<T>(out);
    phi::funcs::SetConstant<Context, T>()(ctx, out, 0);
    return;
  }
  reduce_all = recompute_reduce_all(x, axis.GetData(), reduce_all);
  Reduce<Context, T, funcs::FrobeniusNormFunctor>(
      ctx, x, reduce_all, axis.GetData(), keep_dim, x.dtype(), out);
}

}  // namespace phi
