
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

#include "paddle/phi/kernels/sum_as_kernel.h"

#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/reduce_grad.h"

namespace phi {

template <typename T, typename Context>
void SumAsGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out_grad,
                     bool keep_dim,
                     DenseTensor* x_grad) {
  auto reduce_dim = phi::funcs::GetReduceDims(x, y);
  bool reduce_all = recompute_reduce_all(x, reduce_dim);
  ReduceGradKernel<Context, T, funcs::SumGradFunctor, true>(dev_ctx,
                                                            x,
                                                            paddle::none,
                                                            out_grad,
                                                            reduce_dim,
                                                            keep_dim,
                                                            reduce_all,
                                                            x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(sum_as_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SumAsGradKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   uint8_t,
                   int8_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
