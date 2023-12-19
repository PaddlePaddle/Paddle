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

#include "paddle/phi/kernels/expand_as_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void ExpandAsGradKernel(const Context& context,
                        const DenseTensor& x,
                        const DenseTensor& out_grad,
                        const std::vector<int>& target_shape,
                        DenseTensor* in_grad) {
  auto in_dims = x.dims();
  auto out_dims = out_grad.dims();
  int in_rank = in_dims.size();
  int out_rank = out_dims.size();

  PADDLE_ENFORCE_LE(
      out_rank,
      6,
      errors::InvalidArgument("The rank of the input 'Out@GRAD' for "
                              "expand_as_v2_grad op must be less than or equal "
                              "to 6, but the value received is %d.",
                              out_rank));

  context.template Alloc<T>(in_grad);
  if (in_dims == out_dims) {
    phi::Copy(context, out_grad, context.GetPlace(), false, in_grad);
  } else {
    std::vector<int> reduce_dims = funcs::GetReduceDim(in_dims, out_dims, -1);

    phi::SumKernel<T, Context>(
        context, out_grad, reduce_dims, out_grad.dtype(), false, in_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(expand_as_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ExpandAsGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
