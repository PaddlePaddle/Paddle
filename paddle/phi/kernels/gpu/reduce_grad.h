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

// CUDA and HIP use same api
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"

namespace phi {

template <typename Functor>
void ReduceGrad(const GPUContext& dev_ctx,
                DenseTensor* d_out,
                DenseTensor* d_x,
                DataType out_dtype,
                Functor functor) {
  std::vector<const DenseTensor*> inputs = {d_out};
  std::vector<DenseTensor*> outputs = {d_x};
  PD_VISIT_ALL_TYPES(out_dtype, "BroadcastKernel", ([&] {
                       funcs::BroadcastKernel<data_t>(
                           dev_ctx, inputs, &outputs, functor, 0);
                     }));
}

template <typename OutT, typename Context, typename Functor>
void ReduceGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* x_grad,
                      Functor functor) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto* in_x = &x;
  auto* d_out = &out_grad;
  auto* d_x = x_grad;

  // get reduce_dim and reduce_num for reduce_mean_grad
  int dim_size = in_x->dims().size();
  std::vector<int> reduce_dims =
      funcs::details::GetReduceDim(dims, dim_size, reduce_all);

  auto update_dims = common::vectorize(d_x->dims());
  int reduce_num = 1;
  for (auto i : reduce_dims) {
    reduce_num *= (in_x->dims())[i];
    update_dims[i] = 1;
  }
  // make new tensor
  DenseTensor new_d_out(d_out->dtype());
  new_d_out.ShareDataWith(*d_out);
  new_d_out.Resize(common::make_ddim(update_dims));

  dev_ctx.Alloc(d_x, x.dtype());

  auto pt_d_out = new_d_out;
  auto pt_d_x = *d_x;
  std::vector<const DenseTensor*> inputs = {&pt_d_out};
  std::vector<DenseTensor*> outputs = {&pt_d_x};
  funcs::BroadcastKernel<OutT>(dev_ctx, inputs, &outputs, functor, 0);
}

}  // namespace phi
#endif
