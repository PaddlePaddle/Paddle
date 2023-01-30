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

#include "paddle/phi/kernels/reduce_amin_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/reduce_amin_amax_common.h"

namespace phi {

template <typename T, typename Context>
void ReduceAMinGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& out,
                          const DenseTensor& out_grad,
                          const std::vector<int64_t>& dims,
                          bool keep_dim,
                          bool reduce_all,
                          DenseTensor* x_grad) {
<<<<<<< HEAD
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  ReduceCudaAMaxAMinGrad<T, Context>(
      dev_ctx, x, out, out_grad, dims, keep_dim, reduce_all, x_grad);
}
}  // namespace phi

PD_REGISTER_KERNEL(amin_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceAMinGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
