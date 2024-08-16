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

#include "paddle/phi/kernels/unsqueeze_grad_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"

namespace phi {
template <typename T, typename Context>
void UnsqueezeGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         DenseTensor* dx) {
  // NOTE: [Why not to use x.dims() ?]
  // Because inplace strategy is different between old IR and PIR,
  // we need fix it into x.dims() after cleaning old IR system.
  auto x_dims = dx->dims();
  dev_ctx.template Alloc<T>(dx);
  phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), true, dx);
  dx->Resize(x_dims);
}
}  // namespace phi

PD_REGISTER_KERNEL(unsqueeze_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::UnsqueezeGradKernel,
                   float,
                   double,
                   bool,
                   int,
                   int16_t,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(unsqueeze_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::UnsqueezeGradKernel,
                   float,
                   double,
                   bool,
                   int,
                   int16_t,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(unsqueeze_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::UnsqueezeGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t) {}

#endif
