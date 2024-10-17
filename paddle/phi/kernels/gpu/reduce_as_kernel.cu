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

#include "paddle/phi/kernels/reduce_as_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void ReduceAsKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& target,
                    DenseTensor* out) {
  auto reduce_dim = phi::funcs::GetReduceDims(x, target);
  // std::cerr << "reduce as " << reduce_dim.size() << std::endl;
  // for( auto d : reduce_dim)
  // {
  //   std::cerr << d << " , ";
  // }
  // std::cerr << std::endl;

  dev_ctx.template Alloc<T>(out);
  if (reduce_dim.size() != 0) {
    phi::SumKernel<T, Context>(dev_ctx, x, reduce_dim, out->type(), false, out);
  } else {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), true, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(reduce_as,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceAsKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
