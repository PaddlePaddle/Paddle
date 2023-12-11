// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/copysign_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void CopySignKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  if (x_dims.size() >= y_dims.size()) {
    funcs::ElementwiseCompute<phi::CopySignFunctor<T>, T>(
        dev_ctx, x, y, phi::CopySignFunctor<T>(), out);
  } else {
    funcs::ElementwiseCompute<phi::InverseCopySignFunctor<T>, T>(
        dev_ctx, x, y, phi::InverseCopySignFunctor<T>(), out);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(copysign,
                   CPU,
                   ALL_LAYOUT,
                   phi::CopySignKernel,
                   uint8_t,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
