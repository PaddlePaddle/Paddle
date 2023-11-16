// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/gpu/dirichlet_util.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {
template <typename T>
struct DirichletSampler<GPUContext, T> {
  void operator()(const GPUContext& dev_ctx,
                  const DenseTensor& alpha,
                  DenseTensor* out) {
    auto p_gen = dev_ctx.GetGenerator();
    auto seed_and_offset = p_gen->IncrementOffset(10);  // hard-coded offset
    auto seed = seed_and_offset.first;
    auto offset = seed_and_offset.second;

    // sample from K gamma distributions, where K=alpha.numel()
    DenseTensor gamma_samples;
    gamma_samples.Resize(alpha.dims());
    dev_ctx.template Alloc<T>(&gamma_samples);

    GammaCUDAFunctor<T> gamma_functor(
        alpha.data<T>(), gamma_samples.data<T>(), seed, offset);
    funcs::ForRange<GPUContext> for_range(dev_ctx, out->numel());
    for_range(gamma_functor);

    // normalize them into a simplex, along the last axis
    DenseTensor gamma_sum;
    auto new_shape = gamma_samples.dims();
    new_shape[new_shape.size() - 1] = 1;
    gamma_sum.Resize(new_shape);
    dev_ctx.template Alloc<T>(&gamma_sum);

    phi::SumRawKernel<T, GPUContext>(dev_ctx,
                                     gamma_samples,
                                     {new_shape.size() - 1},
                                     true,
                                     false,
                                     gamma_sum.dtype(),
                                     &gamma_sum);
    phi::DivideKernel<T, GPUContext>(dev_ctx, gamma_samples, gamma_sum, out);
  }
};
}  // namespace phi

PD_REGISTER_KERNEL(dirichlet,
                   GPU,
                   ALL_LAYOUT,
                   phi::Dirichletkernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
