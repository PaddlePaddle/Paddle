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

#include "paddle/phi/kernels/log_softmax_grad_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/log_softmax_functor.h"

namespace phi {

template <typename T, typename Context>
void LogSoftmaxGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& out,
                          const DenseTensor& out_grad,
                          int axis,
                          DenseTensor* x_grad) {
  const int rank = out.dims().size();
  const int canonical_axis = CanonicalAxis(axis, rank);

  dev_ctx.template Alloc<T>(x_grad);
  if (out.numel() != 0) {
    LogSoftmaxGradFunctor<Context, T>()(
        dev_ctx, &out, &out_grad, x_grad, canonical_axis);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(log_softmax_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::LogSoftmaxGradKernel,
                   float,
                   double) {}
