/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/elementwise_grad_base.h"

namespace phi {

// NOTE(dzhwinter): Only used in elementwise_add, elementwise_sub.
// explicit gradient can cut off X, Y, Out from gradient op
// In elementwise_add, elementwise_sub, we use dout as fake X, Y, Out to reuse
// elementwise code.
template <typename T, typename DX_OP, typename DY_OP>
void ElemwiseExplicitGradCompute(const CPUContext& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 const DenseTensor& out,
                                 const DenseTensor& dout,
                                 int axis,
                                 DenseTensor* dx,
                                 DenseTensor* dy,
                                 DX_OP dx_op,
                                 DY_OP dy_op) {
  const DDim& x_dim = x.dims();
  const DDim& y_dim = y.dims();
  if (x.dims() == y.dims()) {
    funcs::ElemwiseGradComputeNoBroadcast<CPUContext, T, DX_OP, DY_OP>(dev_ctx,
                                                                       x_dim,
                                                                       y_dim,
                                                                       dout,
                                                                       dout,
                                                                       out,
                                                                       dout,
                                                                       axis,
                                                                       dx,
                                                                       dy,
                                                                       dx_op,
                                                                       dy_op);
  } else {
    funcs::ElemwiseGradComputeWithBroadcast<T, DX_OP, DY_OP>(dev_ctx,
                                                             x_dim,
                                                             y_dim,
                                                             dout,
                                                             dout,
                                                             out,
                                                             dout,
                                                             axis,
                                                             dx,
                                                             dy,
                                                             dx_op,
                                                             dy_op);
  }
}

/*
******************************
    Add Grad
******************************
*/
template <typename T>
struct IdentityGrad {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value>::type
ElementwiseAddGrad(const CPUContext& ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   const DenseTensor& out,
                   const DenseTensor& dout,
                   DenseTensor* dx,
                   DenseTensor* dy,
                   int axis = -1) {
  auto blas = phi::funcs::GetBlas<CPUContext, T>(ctx);
  if (dx) {
    blas.VCOPY(dout.numel(), dout.data<T>(), ctx.template Alloc<T>(dx));
  }

  if (dy) {
    blas.VCOPY(dout.numel(), dout.data<T>(), ctx.template Alloc<T>(dy));
  }
}

template <typename T>
typename std::enable_if<!std::is_floating_point<T>::value>::type
ElementwiseAddGrad(const CPUContext& ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   const DenseTensor& out,
                   const DenseTensor& dout,
                   DenseTensor* dx,
                   DenseTensor* dy,
                   int axis = -1) {
  ElemwiseExplicitGradCompute<T, IdentityGrad<T>, IdentityGrad<T>>(
      ctx, x, y, out, dout, axis, dx, dy, IdentityGrad<T>(), IdentityGrad<T>());
}

/*
******************************
    Sub Grad
******************************
*/

template <typename T>
struct SubGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename T>
struct SubGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return -dout; }
};

template <typename T>
void ElementwiseSubGrad(const CPUContext& ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& out,
                        const DenseTensor& dout,
                        DenseTensor* dx,
                        DenseTensor* dy,
                        int axis = -1) {
  ElemwiseExplicitGradCompute<T, SubGradDX<T>, SubGradDY<T>>(
      ctx, x, y, out, dout, axis, dx, dy, SubGradDX<T>(), SubGradDY<T>());
}

}  // namespace phi
