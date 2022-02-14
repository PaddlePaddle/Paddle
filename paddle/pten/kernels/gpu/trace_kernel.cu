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

#include "paddle/pten/kernels/trace_kernel.h"

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/funcs/diagonal.h"
#include "paddle/pten/kernels/gpu/reduce.h"

namespace pten {

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
    auto stream = ctx.stream();
    std::vector<int> reduce_dims;
    reduce_dims.push_back(out->dims().size());
    kernels::TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
        ctx, diag, out, kps::IdentityFunctor<T>(), reduce_dims, stream);
  } else {
    pten::funcs::SetConstant<Context, T> functor;
    functor(ctx, out, static_cast<T>(0));
  }
}

}  // namespace pten

PT_REGISTER_KERNEL(trace,
                   GPU,
                   ALL_LAYOUT,
                   pten::TraceKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   pten::dtype::float16,
                   pten::dtype::complex<float>,
                   pten::dtype::complex<double>) {}
