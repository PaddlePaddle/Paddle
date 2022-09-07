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
  using MPType = typename kps::details::MPTypeTrait<T>::Type;
  auto out_dtype = x.dtype();
  auto* in_x = &x;
  auto* d_out = &out_grad;
  auto* d_x = x_grad;

  // get reduce_dim and reduce_num for reduce_mean_grad
  int dim_size = in_x->dims().size();
  if (dims.size() == 0) {
    reduce_all = true;
  }
  std::vector<int> reduce_dims =
      funcs::details::GetReduceDim(dims.GetData(), dim_size, reduce_all);

  auto update_dims = vectorize(d_x->dims());
  int reduce_num = 1;
  for (auto i : reduce_dims) {
    reduce_num *= (in_x->dims())[i];
    update_dims[i] = 1;
  }
  // make new tensor
  DenseTensor new_d_out(d_out->dtype());
  new_d_out.ShareDataWith(*d_out);
  new_d_out.Resize(phi::make_ddim(update_dims));

  dev_ctx.Alloc(d_x, x.dtype());
  auto pt_out_dtype = x.dtype();
  auto pt_d_out = new_d_out;
  auto pt_d_x = *d_x;
  std::vector<const DenseTensor*> inputs = {&pt_d_out};
  std::vector<DenseTensor*> outputs = {&pt_d_x};
  phi::ReduceGrad<T, kps::IdentityFunctor<T, MPType>>(
      dev_ctx,
      &pt_d_out,
      &pt_d_x,
      pt_out_dtype,
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
