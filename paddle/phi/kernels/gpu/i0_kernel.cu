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

#include "paddle/phi/kernels/i0_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/abs_util.h"
#include "paddle/phi/kernels/impl/bessel_kernel_cuda_impl.h"

namespace phi {

template <typename T, typename Context>
void I0Kernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  ctx.template Alloc<T>(out);
  auto functor = CudaI0Functor<T>();
  funcs::TensorContainer ten_con(&x, out);
  funcs::test_func(ctx.stream(), ten_con, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(i0, GPU, ALL_LAYOUT, phi::I0Kernel, float, double) {}
