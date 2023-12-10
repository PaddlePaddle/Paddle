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

#include "paddle/phi/kernels/standard_gamma_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/gpu/dirichlet_util.h"

namespace phi {
template <typename T, typename Context>
void StandardGammaKernel(const Context& dev_ctx,
                         const DenseTensor& alpha,
                         DenseTensor* out) {
  auto p_gen = dev_ctx.GetGenerator();
  auto seed_and_offset = p_gen->IncrementOffset(10);  // hard-coded offset
  auto seed = seed_and_offset.first;
  auto offset = seed_and_offset.second;

  dev_ctx.template Alloc<T>(out);

  GammaCUDAFunctor<T> gamma_functor(
      alpha.data<T>(), out->data<T>(), seed, offset);
  funcs::ForRange<GPUContext> for_range(dev_ctx, out->numel());
  for_range(gamma_functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(standard_gamma,
                   GPU,
                   ALL_LAYOUT,
                   phi::StandardGammaKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
