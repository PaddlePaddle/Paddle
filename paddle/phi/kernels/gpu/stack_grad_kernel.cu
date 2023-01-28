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

#include "paddle/phi/kernels/stack_grad_kernel.h"

#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/stack_and_unstack.h"

namespace phi {

template <typename T, typename Context>
void StackGradKernel(const Context& ctx,
                     const DenseTensor& out_grad,
                     int axis,
                     std::vector<DenseTensor*> x_grad) {
  if (axis < 0) axis += out_grad.dims().size();

  int64_t split_dim = out_grad.dims()[axis];
  PADDLE_ENFORCE_EQ(
      split_dim,
      x_grad.size(),
      phi::errors::InvalidArgument(
          "Output x_grad's size should be equal to the split_dim, but"
          " received split_dim is:%d x_grad's size is:%d.",
          split_dim,
          x_grad.size()));

  funcs::UnStackRawKernel<T, Context>(ctx, out_grad, axis, &x_grad);
}

template <typename T, typename Context>
void UnStackKernel(const Context& ctx,
                   const DenseTensor& x,
                   int axis,
                   int num,
                   std::vector<DenseTensor*> outs) {
  if (x.numel() == 0) return;
  if (axis < 0) axis += x.dims().size();

  int64_t split_dim = x.dims()[axis];
  PADDLE_ENFORCE_EQ(
      split_dim,
      outs.size(),
      phi::errors::InvalidArgument(
          "Output outs's size should be equal to the split_dim, but"
          " received split_dim is:%d outs's size is:%d.",
          split_dim,
          outs.size()));

  UnStackRawKernel<T, Context>(ctx, x, axis, &outs);
}

}  // namespace phi

PD_REGISTER_KERNEL(stack_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::StackGradKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
