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

#include "paddle/phi/kernels/expand_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void ExpandGradKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const IntArray& shape,
                      DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);
  if (x_grad->dims() == out_grad.dims()) {
    phi::Copy(ctx, out_grad, ctx.GetPlace(), false, x_grad);
  } else {
    std::vector<int> reduce_dims =
        funcs::GetReduceDim(x_grad->dims(), out_grad.dims(), -1);
    phi::SumKernel<T, Context>(
        ctx, out_grad, reduce_dims, out_grad.dtype(), false, x_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(expand_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ExpandGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
