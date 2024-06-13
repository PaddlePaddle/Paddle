// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/reduce_as_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/gpu/reduce_grad.h"

namespace phi {

template <typename T, typename Context>
void ReduceAsGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& target,
                        const DenseTensor& out_grad,
                        DenseTensor* x_grad) {
  auto reduce_dim = phi::funcs::GetReduceDims(x, target);
  dev_ctx.Alloc(x_grad, x.dtype());

  if (reduce_dim.size() == 0) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    return;
  }
  auto update_dims = common::vectorize(x.dims());
  for (auto i : reduce_dim) {
    update_dims[i] = 1;
  }

  DenseTensor new_out_grad(out_grad.type());
  new_out_grad.ShareDataWith(out_grad);
  new_out_grad.Resize(common::make_ddim(update_dims));

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  phi::ReduceGrad<phi::kps::IdentityFunctor<T, MPType>>(
      dev_ctx,
      &new_out_grad,
      x_grad,
      out_grad.dtype(),
      phi::kps::IdentityFunctor<T, MPType>());
}

}  // namespace phi

PD_REGISTER_KERNEL(reduce_as_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceAsGradKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
