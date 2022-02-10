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

#include "paddle/pten/kernels/elementwise_grad_kernel.h"

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/copy_kernel.h"
#include "paddle/pten/kernels/funcs/elementwise_functor.h"
#include "paddle/pten/kernels/gpu/elementwise.h"
#include "paddle/pten/kernels/impl/elementwise_grad_kernel_impl.h"

namespace pten {

template <typename T>
void AddGradFunc(const GPUContext& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 const DenseTensor& out,
                 const DenseTensor& dout,
                 DenseTensor* dx,
                 DenseTensor* dy,
                 int axis = -1) {
  if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
    elementwise_add_grad<T>(dev_ctx, x, y, out, dout, dx, dy);
  } else {
    default_elementwise_add_grad<T>(dev_ctx, x, y, out, dout, dx, dy, axis);
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
  pten::AddGradImpl<T>(dev_ctx, x, y, dout, axis, dx, dy, AddGradFunc<T>);
}

template <typename T, typename Context>
void AddDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& y,
                         paddle::optional<const DenseTensor&> ddx,
                         paddle::optional<const DenseTensor&> ddy,
                         const DenseTensor& dout,
                         int axis,
                         DenseTensor* ddout) {
  pten::AddDoubleGradImpl<T>(
      dev_ctx,
      y,
      ddx,
      ddy,
      dout,
      axis,
      ddout,
      ElementwiseCompute<funcs::AddFunctor<T>, T>,
      ElementwiseCompute<funcs::InverseAddFunctor<T>, T>);
}

template <typename T, typename Context>
void AddTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& ddx,
                         const DenseTensor& ddy,
                         const DenseTensor& d_ddout,
                         int axis,
                         DenseTensor* d_ddx,
                         DenseTensor* d_ddy) {
  pten::AddGradImpl<T>(
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
  if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
    elementwise_sub_grad<T>(dev_ctx, x, y, *out, dout, dx, dy);
  } else {
    default_elementwise_sub_grad<T>(dev_ctx, x, y, *out, dout, dx, dy, axis);
  }
}

template <typename T, typename Context>
void SubtractDoubleGradKernel(const Context& dev_ctx,
                              const DenseTensor& y,
                              paddle::optional<const DenseTensor&> ddx,
                              paddle::optional<const DenseTensor&> ddy,
                              const DenseTensor& dout,
                              int axis,
                              DenseTensor* ddout) {
  pten::SubtractDoubleGradImpl<T>(
      dev_ctx,
      y,
      ddx,
      ddy,
      dout,
      axis,
      ddout,
      ElementwiseCompute<funcs::SubtractFunctor<T>, T>);
}

}  // namespace pten

PT_REGISTER_KERNEL(add_grad,
                   GPU,
                   ALL_LAYOUT,
                   pten::AddGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   paddle::platform::float16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}

PT_REGISTER_KERNEL(add_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   pten::AddDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   paddle::platform::float16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}

PT_REGISTER_KERNEL(add_triple_grad,
                   GPU,
                   ALL_LAYOUT,
                   pten::AddTripleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   paddle::platform::float16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}

PT_REGISTER_KERNEL(subtract_grad,
                   GPU,
                   ALL_LAYOUT,
                   pten::SubtractGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   paddle::platform::float16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}

PT_REGISTER_KERNEL(subtract_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   pten::SubtractDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   paddle::platform::float16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}
