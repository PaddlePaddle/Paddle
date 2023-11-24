/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/igamma_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T, typename Context>
void IgammaKernel(const Context& ctx,
                  const DenseTensor& x,
                  const DenseTensor& a,
                  DenseTensor* out) {
    const int64_t size = x.numel();
    const T* x_data = x.data<T>();
    const T* a_data = a.data<T>();
    T* out_data = ctx.template Alloc<T>(out);

    phi::funcs::ForRange<Context> for_range(ctx, size);
    IgammaFunctor<T> functor(x_data, a_data, out_data, size);
    for_range(functor);

}

}  // namespace phi

PD_REGISTER_KERNEL(igamma,
                   CPU,
                   ALL_LAYOUT,
                   phi::IgammaKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
