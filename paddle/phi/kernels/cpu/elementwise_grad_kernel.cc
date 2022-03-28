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

#include "paddle/phi/kernels/elementwise_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
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
                         paddle::optional<const DenseTensor&> ddx,
                         paddle::optional<const DenseTensor&> ddy,
                         const DenseTensor& dout,
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

template <typename T, typename Context>
void DivideGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& out,
                      const DenseTensor& dout,
                      int axis,
                      DenseTensor* dx,
                      DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  phi::funcs::ElemwiseGradCompute<Context, T, DivGradDX<T>, DivGradDY<T>>(
      dev_ctx, x, y, out, dout, axis, dx, dy, DivGradDX<T>(), DivGradDY<T>());
}

template <typename T, typename Context>
void MultiplyGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& dout,
                        int axis,
                        DenseTensor* dx,
                        DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  auto* out = &dout;  // out is not necessary
  phi::funcs::ElemwiseGradCompute<Context, T, MulGradDX<T>, MulGradDY<T>>(
      dev_ctx, x, y, *out, dout, axis, dx, dy, MulGradDX<T>(), MulGradDY<T>());
}

template <typename T, typename Context>
void MaximumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       int axis,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  phi::funcs::ElemwiseGradCompute<Context, T, MaxGradDx<T>, MaxGradDy<T>>(
      dev_ctx, x, y, dout, dout, axis, dx, dy, MaxGradDx<T>(), MaxGradDy<T>());
}

template <typename T, typename Context>
void MinimumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       int axis,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  phi::funcs::ElemwiseGradCompute<Context, T, MinGradDx<T>, MinGradDy<T>>(
      dev_ctx, x, y, dout, dout, axis, dx, dy, MinGradDx<T>(), MinGradDy<T>());
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

PD_REGISTER_KERNEL(divide_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::DivideGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(divide_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::DivideDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(multiply_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MultiplyGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(multiply_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MultiplyDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(multiply_triple_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MultiplyTripleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(fmax_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwiseFMaxGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(fmin_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwiseFMinGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(maximum_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaximumGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(minimum_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MinimumGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
