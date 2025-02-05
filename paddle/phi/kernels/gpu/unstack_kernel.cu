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

#include "paddle/phi/kernels/unstack_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/stack_and_unstack.h"

namespace phi {

template <typename T, typename Context>
void UnStackKernel(const Context& ctx,
                   const DenseTensor& x,
                   int axis,
                   int num,
                   std::vector<DenseTensor*> outs) {
  if (axis < 0) axis += x.dims().size();

  int64_t split_dim = x.dims()[axis];
  PADDLE_ENFORCE_EQ(
      split_dim,
      outs.size(),
      common::errors::InvalidArgument(
          "Output outs's size should be equal to the split_dim, but"
          " received split_dim is:%d outs's size is:%d.",
          split_dim,
          outs.size()));

  funcs::UnStackRawKernel<T, Context>(ctx, x, axis, &outs);
}

}  // namespace phi

PD_REGISTER_KERNEL(unstack,
                   GPU,
                   ALL_LAYOUT,
                   phi::UnStackKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
