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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/gpu/elementwise_grad.h"
#include "paddle/phi/kernels/impl/elementwise_grad_kernel_impl.h"

namespace phi {

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
    ElementwiseAddGrad<T>(dev_ctx, x, y, out, dout, dx, dy);
  } else {
    DefaultElementwiseAddGrad<T>(dev_ctx, x, y, out, dout, dx, dy, axis);
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
                         paddle::optional<const DenseTensor&> ddx,
                         paddle::optional<const DenseTensor&> ddy,
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
                   GPU,
                   ALL_LAYOUT,
                   phi::AddGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(add_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AddDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(add_triple_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AddTripleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
