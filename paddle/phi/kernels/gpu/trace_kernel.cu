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

#include "paddle/phi/kernels/trace_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/diagonal.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void TraceKernel(const Context& ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out) {
  T* out_data = ctx.template Alloc<T>(out);
  auto diag = funcs::Diagonal<T, Context>(ctx, &x, offset, axis1, axis2);
  if (diag.numel() > 0) {
    std::vector<int> reduce_dims;
    // Adapt to 0D output
    auto out_dim_size = out->dims().size();
    if (out_dim_size == 0) out_dim_size = 1;
    reduce_dims.push_back(out_dim_size);
    phi::SumKernel<T, Context>(
        ctx, diag, reduce_dims, diag.dtype(), false, out);
  } else {
    phi::funcs::SetConstant<Context, T> functor;
    functor(ctx, out, static_cast<T>(0));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(trace,
                   GPU,
                   ALL_LAYOUT,
                   phi::TraceKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
