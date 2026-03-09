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

#include "paddle/phi/kernels/reduce_nansum_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/impl/reduce_grad.h"

namespace phi {

template <typename T, typename Context>
void NansumGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const IntArray& dims,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* x_grad) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }

  // Step 1: broadcast out_grad to x_grad shape (same as sum_grad)
  ReduceGradKernel<Context, T, funcs::SumGradFunctor, true>(dev_ctx,
                                                            x,
                                                            paddle::none,
                                                            out_grad,
                                                            dims.GetData(),
                                                            keep_dim,
                                                            reduce_all,
                                                            x_grad);

  // Step 2: zero out gradient where x is NaN
  const T* x_data = x.data<T>();
  T* x_grad_data = x_grad->data<T>();
  int64_t numel = x.numel();
  for (int64_t i = 0; i < numel; ++i) {
    if (x_data[i] != x_data[i]) {
      x_grad_data[i] = static_cast<T>(0);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(nansum_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::NansumGradKernel,
                   bool,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
