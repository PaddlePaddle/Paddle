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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/compare_functors.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void ReduceCudaAMaxAMinGrad(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& out,
                            const DenseTensor& out_grad,
                            const std::vector<int64_t>& dims,
                            bool keep_dim,
                            bool reduce_all,
                            DenseTensor* x_grad) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto* in_x = &x;
  auto* out_y = &out;
  auto* d_out = &out_grad;
  auto* d_x = x_grad;
  // get reduce_dim and reduce_num for reduce_mean_grad
  int dim_size = in_x->dims().size();
  auto reduce_dims = funcs::details::GetReduceDim(dims, dim_size, reduce_all);
  auto update_dims = common::vectorize(d_x->dims());
  int reduce_num = 1;
  for (auto i : reduce_dims) {
    reduce_num *= (in_x->dims())[i];
    update_dims[i] = 1;
  }

  // make new tensor reduce_out
  phi::DenseTensor new_y(out_y->type());
  new_y.ShareDataWith(*out_y);
  new_y.Resize(common::make_ddim(update_dims));

  // make new tensor d_out
  phi::DenseTensor new_dout(d_out->type());
  new_dout.ShareDataWith(*d_out);
  new_dout.Resize(common::make_ddim(update_dims));
  dev_ctx.Alloc(d_x, d_out->dtype());

  auto new_in = std::make_unique<phi::DenseTensor>(*in_x);
  auto new_in_tensor = new_in.get();

  auto new_dx = std::make_unique<phi::DenseTensor>(*d_x);
  auto new_dx_tensor = new_dx.get();

  // make equal_out
  phi::DenseTensor* equal_out = new phi::DenseTensor();
  equal_out->Resize(in_x->dims());
  dev_ctx.template Alloc<T>(equal_out);
  auto equal_out_tensor = *equal_out;

  // make new tensor equal_count
  phi::DenseTensor* equal_count = new phi::DenseTensor();
  equal_count->Resize(common::make_ddim(update_dims));
  dev_ctx.template Alloc<T>(equal_count);

  // compute
  // 1. equal_out = Equal(x, y)
  std::vector<const phi::DenseTensor*> equal_inputs = {&new_y, new_in_tensor};
  std::vector<phi::DenseTensor*> equal_outputs = {&equal_out_tensor};
  funcs::BroadcastKernel<T>(
      dev_ctx, equal_inputs, &equal_outputs, funcs::EqualFunctor<T>(), 0);
  // 2. equal_count = reduceSum(equal_out)
  phi::SumKernel<T, Context>(dev_ctx,
                             equal_out_tensor,
                             reduce_dims,
                             equal_out_tensor.dtype(),
                             false,
                             equal_count);
  // 3. dx = dout * 1
  phi::MultiplyKernel<T, Context>(
      dev_ctx, new_dout, equal_out_tensor, &equal_out_tensor);

  // 4. dx = Div(dx, equal_out)
  phi::DivideKernel<T, Context>(
      dev_ctx, equal_out_tensor, *equal_count, new_dx_tensor);
  delete equal_out;
  delete equal_count;
}
}  // namespace phi
