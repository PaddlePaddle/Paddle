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

#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/gpu/reduce_grad.h"

namespace phi {

template <typename T, typename Context>
void ReduceSumGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out_grad,
                         const IntArray& dims,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* x_grad) {
  // get reduce_dim for reduce_mean_grad
  int dim_size = x.dims().size();
  if (dims.size() == 0) {
    reduce_all = true;
  }
  std::vector<int> reduce_dims =
      funcs::details::GetReduceDim(dims.GetData(), dim_size, reduce_all);

  auto update_dims = vectorize(x.dims());
  for (auto i : reduce_dims) {
    update_dims[i] = 1;
  }

  // make new tensor
  DenseTensor new_out_grad(out_grad.dtype());
  new_out_grad.ShareDataWith(out_grad);
  new_out_grad.Resize(phi::make_ddim(update_dims));

  // call ReduceGrad
  dev_ctx.Alloc(x_grad, x.dtype());
  using MPType = typename kps::details::MPTypeTrait<T>::Type;
  phi::ReduceGrad<T, kps::IdentityFunctor<T, MPType>>(
      dev_ctx,
      &new_out_grad,
      x_grad,
      x.dtype(),
      kps::IdentityFunctor<T, MPType>());
}

}  // namespace phi

PD_REGISTER_KERNEL(sum_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceSumGradKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
