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

#include "paddle/phi/kernels/elementwise_maximum_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MaximumRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      int axis,
                      DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::MaximumFunctor<T>, T>(
      dev_ctx, x, y, axis, funcs::MaximumFunctor<T>(), out);
}

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MaximumRawKernel<T>(dev_ctx, x, y, axis, out);
}

}  // namespace phi

// NOTE(chenweihang): using bfloat16 will cause redefine with xpu bfloat16
// using bfloat16 = ::phi::dtype::bfloat16;

PD_REGISTER_KERNEL(maximum_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaximumRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(maximum,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaximumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
