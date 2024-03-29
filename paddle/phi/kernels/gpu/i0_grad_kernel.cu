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

#include "paddle/phi/kernels/i0_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/impl/bessel_grad_kernel_cuda_impl.h"

namespace phi {

template <typename T, typename Context>
void I0GradKernel(const Context& ctx,
                  const DenseTensor& x,
                  const DenseTensor& out_grad,
                  DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);
  std::vector<const DenseTensor*> ins = {&x, &out_grad};
  std::vector<DenseTensor*> outs = {x_grad};
  auto functor = CudaI0GradFunctor<T>();
  phi::funcs::ElementwiseKernel<T>(ctx, ins, &outs, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(i0_grad, GPU, ALL_LAYOUT, phi::I0GradKernel, float, double) {
}
