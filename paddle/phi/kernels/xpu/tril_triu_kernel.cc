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

#include "paddle/phi/kernels/tril_triu_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TrilTriuKernel(const Context& ctx,
                    const DenseTensor& x,
                    int diagonal,
                    bool lower,
                    DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  ctx.template Alloc<T>(out);
  auto xshape = common::vectorize<int>(x.dims());
  int r = 0;
  if (lower) {
    r = xpu::tril(ctx.x_context(),
                  reinterpret_cast<const XPUType*>(x.data<T>()),
                  reinterpret_cast<XPUType*>(out->data<T>()),
                  xshape,
                  diagonal);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "tril_op");
  } else {
    r = xpu::triu(ctx.x_context(),
                  reinterpret_cast<const XPUType*>(x.data<T>()),
                  reinterpret_cast<XPUType*>(out->data<T>()),
                  xshape,
                  diagonal);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "triu_op");
  }
}

template <typename T, typename Context>
void TrilKernel(const Context& ctx,
                const DenseTensor& x,
                int diagonal,
                DenseTensor* out) {
  TrilTriuKernel<T, Context>(ctx, x, diagonal, true, out);
}

template <typename T, typename Context>
void TriuKernel(const Context& ctx,
                const DenseTensor& x,
                int diagonal,
                DenseTensor* out) {
  TrilTriuKernel<T, Context>(ctx, x, diagonal, false, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(tril_triu,
                   XPU,
                   ALL_LAYOUT,
                   phi::TrilTriuKernel,
                   int,
                   int64_t,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool) {}
PD_REGISTER_KERNEL(tril,
                   XPU,
                   ALL_LAYOUT,
                   phi::TrilKernel,
                   int,
                   int64_t,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool) {}
PD_REGISTER_KERNEL(triu,
                   XPU,
                   ALL_LAYOUT,
                   phi::TriuKernel,
                   int,
                   int64_t,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool) {}
