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

#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/impl/reduce_grad.h"
namespace phi {

template <typename T, typename Context>
void ReduceSumGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out_grad,
                         const IntArray& dims,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* x_grad) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  ReduceGradKernel<Context, T, funcs::SumGradFunctor, true>(dev_ctx,
                                                            x,
                                                            paddle::none,
                                                            out_grad,
                                                            dims.GetData(),
                                                            keep_dim,
                                                            reduce_all,
                                                            x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(sum_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ReduceSumGradKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
