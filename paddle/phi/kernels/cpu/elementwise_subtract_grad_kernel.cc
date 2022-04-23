//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/elementwise_subtract_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/elementwise_grad.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/impl/elementwise_grad_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void SubtractGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& dout,
                        int axis,
                        DenseTensor* dx,
                        DenseTensor* dy) {
  // skip out
  auto* out = &dout;
  ElementwiseSubGrad<T>(dev_ctx, x, y, *out, dout, dx, dy, axis);
}

template <typename T, typename Context>
void SubtractDoubleGradKernel(const Context& dev_ctx,
                              const DenseTensor& y,
                              paddle::optional<const DenseTensor&> ddx,
                              paddle::optional<const DenseTensor&> ddy,
                              const DenseTensor& dout,
                              int axis,
                              DenseTensor* ddout) {
  phi::SubtractDoubleGradImpl<T>(dev_ctx, y, ddx, ddy, dout, axis, ddout);
}

}  // namespace phi

PD_REGISTER_KERNEL(subtract_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SubtractGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(subtract_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SubtractDoubleGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
