//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/flatten_grad_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"

namespace phi {

template <typename T, typename Context>
void FlattenGradKernel(const Context& dev_ctx,
                       const DenseTensor& out_grad,
                       const DenseTensor& xshape,
                       DenseTensor* x_grad) {
  auto xshape_dims = xshape.dims();
  dev_ctx.Alloc(x_grad, out_grad.dtype());
  auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());
  phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
  x_grad->Resize(x_dims);
}

}  // namespace phi

PD_REGISTER_KERNEL(flatten_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::FlattenGradKernel,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(flatten_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlattenGradKernel,
                   float,
                   phi::dtype::float16,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}

#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(flatten_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlattenGradKernel,
                   float,
                   phi::dtype::float16,
                   int8_t,
                   int,
                   int64_t) {}

#endif
