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

#include "paddle/phi/kernels/reduce_mean_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"

namespace phi {

template <typename T, typename Context>
void ReduceMeanGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& out_grad,
                          const IntArray& dims,
                          bool keep_dim,
                          bool reduce_all,
                          DenseTensor* x_grad) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  // get reduce_dim and reduce_num for reduce_mean_grad
  int dim_size = x.dims().size();
  std::vector<int> reduce_dims =
      funcs::details::GetReduceDim(dims.GetData(), dim_size, reduce_all);

  auto update_dims = vectorize(x.dims());
  int reduce_num = 1;
  for (auto i : reduce_dims) {
    reduce_num *= (x.dims())[i];
    update_dims[i] = 1;
  }

  // make new tensor
  DenseTensor new_out_grad(out_grad.dtype());
  new_out_grad.ShareDataWith(out_grad);
  new_out_grad.Resize(phi::make_ddim(update_dims));

  // call BroadcastKernel
  dev_ctx.Alloc(x_grad, x.dtype());
  std::vector<const DenseTensor*> inputs = {&new_out_grad};
  std::vector<DenseTensor*> outputs = {x_grad};

  using MPType = typename kps::details::MPTypeTrait<T>::Type;
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, kps::DivideFunctor<T, MPType>(reduce_num), 0);
}

}  // namespace phi

PD_REGISTER_KERNEL(mean_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceMeanGradKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
