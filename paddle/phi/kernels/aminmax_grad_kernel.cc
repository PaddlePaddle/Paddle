// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/aminmax_grad_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/reduce_amax_grad_kernel.h"
#include "paddle/phi/kernels/reduce_amin_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void AMinMaxGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& min,
                       const DenseTensor& max,
                       const DenseTensor& min_grad,
                       const DenseTensor& max_grad,
                       const std::vector<int64_t>& dims,
                       bool keep_dim,
                       bool reduce_all,
                       DenseTensor* x_grad) {
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  reduce_all = recompute_reduce_all(x, dims, reduce_all);

  // Compute amax grad contribution into x_grad
  ReduceAMaxGradKernel<T, Context>(
      dev_ctx, x, max, max_grad, dims, keep_dim, reduce_all, x_grad);

  // Compute amin grad contribution into a temporary tensor
  DenseTensor amin_x_grad;
  amin_x_grad.Resize(x_grad->dims());
  dev_ctx.template Alloc<T>(&amin_x_grad);
  ReduceAMinGradKernel<T, Context>(
      dev_ctx, x, min, min_grad, dims, keep_dim, reduce_all, &amin_x_grad);

  // x_grad = amax_grad_result + amin_grad_result
  Add<T, Context>(dev_ctx, *x_grad, amin_x_grad, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(aminmax_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::AMinMaxGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(aminmax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AMinMaxGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
#endif
