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

#include "paddle/phi/kernels/reduce_max_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/compare_functors.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"

namespace phi {

template <typename T, typename Context>
void ReduceMaxGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out,
                         const DenseTensor& out_grad,
                         const IntArray& dims,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* x_grad) {
  dev_ctx.Alloc(x_grad, x.dtype());
  reduce_all = recompute_reduce_all(x, dims, reduce_all);

  // get reduce_dim
  int dim_size = x.dims().size();
  auto reduce_dims =
      funcs::details::GetReduceDim(dims.GetData(), dim_size, reduce_all);
  auto update_dims = vectorize(x.dims());
  for (auto i : reduce_dims) {
    update_dims[i] = 1;
  }

  // make new tensor of out and out_grad
  phi::DenseTensor new_out(out.type());
  new_out.ShareDataWith(out);
  new_out.Resize(phi::make_ddim(update_dims));

  phi::DenseTensor new_out_grad(out_grad.type());
  new_out_grad.ShareDataWith(out_grad);
  new_out_grad.Resize(phi::make_ddim(update_dims));

  // make equal_out
  phi::DenseTensor* equal_out = new phi::DenseTensor();
  equal_out->Resize(x.dims());
  dev_ctx.template Alloc<T>(equal_out);

  // compute
  // 1. equal_out = Equal(x, y)
  std::vector<const phi::DenseTensor*> equal_inputs = {&new_out, &x};
  std::vector<phi::DenseTensor*> equal_outputs = {equal_out};
  funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
      dev_ctx, equal_inputs, &equal_outputs, 0, funcs::EqualFunctor<T>());

  // 2. dx = dout * 1
  std::vector<const phi::DenseTensor*> mul_inputs = {&new_out_grad, equal_out};
  std::vector<phi::DenseTensor*> mul_outputs = {x_grad};
  funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
      dev_ctx, mul_inputs, &mul_outputs, 0, funcs::MultiplyFunctor<T>());
  delete equal_out;
}
}  // namespace phi

PD_REGISTER_KERNEL(max_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceMaxGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
