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

#include "paddle/phi/kernels/elementwise_add_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/elementwise_grad.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/impl/elementwise_grad_kernel_impl.h"

namespace phi {

template <typename T>
void AddGradFunc(const CPUContext& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 const DenseTensor& out,
                 const DenseTensor& dout,
                 DenseTensor* dx,
                 DenseTensor* dy,
                 int axis = -1) {
  if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
    ElementwiseAddGrad<T>(dev_ctx, x, y, out, dout, dx, dy);
  } else {
    ElemwiseExplicitGradCompute<T, IdentityGrad<T>, IdentityGrad<T>>(
        dev_ctx,
        x,
        y,
        out,
        dout,
        axis,
        dx,
        dy,
        IdentityGrad<T>(),
        IdentityGrad<T>());
  }
}

template <typename T, typename Context>
void AddGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   const DenseTensor& dout,
                   int axis,
                   DenseTensor* dx,
                   DenseTensor* dy) {
  phi::AddGradImpl<T>(dev_ctx, x, y, dout, axis, dx, dy, AddGradFunc<T>);
}

template <typename T, typename Context>
void AddDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& y,
                         const DenseTensor& dout,
                         const paddle::optional<DenseTensor>& ddx,
                         const paddle::optional<DenseTensor>& ddy,
                         int axis,
                         DenseTensor* ddout) {
  phi::AddDoubleGradImpl<T>(dev_ctx, y, ddx, ddy, dout, axis, ddout);
}

template <typename T, typename Context>
void AddTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& ddx,
                         const DenseTensor& ddy,
                         const DenseTensor& d_ddout,
                         int axis,
                         DenseTensor* d_ddx,
                         DenseTensor* d_ddy) {
  phi::AddGradImpl<T>(
      dev_ctx, ddx, ddy, d_ddout, axis, d_ddx, d_ddy, AddGradFunc<T>);
}

}  // namespace phi

PD_REGISTER_KERNEL(add_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   bool,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(add_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddDoubleGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(add_triple_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddTripleGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
