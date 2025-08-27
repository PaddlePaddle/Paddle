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

#include "paddle/phi/kernels/p_norm_grad_kernel.h"

#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/reduce_grad_functions.h"
#include "paddle/phi/kernels/reduce_amax_grad_kernel.h"
#include "paddle/phi/kernels/sign_kernel.h"

namespace phi {

template <typename T>
struct PNormGradFunctor {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  HOSTDEVICE explicit inline PNormGradFunctor(float porder, float eps) {
    this->porder = static_cast<MT>(porder - 1.0f);
    this->eps = static_cast<MT>(eps);
  }

  template <typename Context,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const Context& place,
                  X* x,
                  Y* y,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size) {
    auto unstable_term =
        (*x).abs().template cast<MT>().pow(this->porder).template cast<T>();
    auto mask = (*x) == x->constant(static_cast<T>(0));
    auto stable_term =
        mask.select(x->constant(static_cast<T>(0)), unstable_term);
    auto self_scaled = (*x).sign() * stable_term;
    auto norm_term =
        (*y).template cast<MT>().pow(-this->porder).template cast<T>();
    dx->device(place) =
        self_scaled * dy->broadcast(dim) * norm_term.broadcast(dim);
  }

  MT porder;
  MT eps;
};

template <typename T, typename Context>
void PNormGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out,
                     const DenseTensor& out_grad,
                     float porder,
                     int axis,
                     float epsilon,
                     bool keepdim,
                     bool asvector,
                     DenseTensor* x_grad) {
  auto* in_x = &x;
  auto* in_norm = &out;
  auto* in_norm_dy = &out_grad;
  auto* out_dx = x_grad;
  dev_ctx.template Alloc<T>(out_dx);

  auto xdim = in_x->dims();
  bool reduce_all = (in_norm->numel() == 1);
  if (axis < 0) {
    axis = xdim.size() + axis;
  }
  const std::vector<int> dims = {axis};

  if (porder == 0) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, out_dx, static_cast<T>(0));
  } else if (porder == INFINITY || porder == -INFINITY) {
    std::vector<int64_t> dims_for_amax;
    if (reduce_all) {
      dims_for_amax.resize(xdim.size());
      for (int i = 0; i < xdim.size(); ++i) dims_for_amax[i] = i;
    } else {
      dims_for_amax.push_back(axis);
    }

    DenseTensor x_abs;
    x_abs.Resize(in_x->dims());
    dev_ctx.template Alloc<T>(&x_abs);
    phi::AbsKernel<T, Context>(dev_ctx, *in_x, &x_abs);

    DenseTensor amax_grad_out;
    amax_grad_out.Resize(in_x->dims());
    dev_ctx.template Alloc<T>(&amax_grad_out);
    phi::ReduceAMaxGradKernel<T, Context>(dev_ctx,
                                          x_abs,
                                          *in_norm,
                                          *in_norm_dy,
                                          dims_for_amax,
                                          keepdim,
                                          reduce_all,
                                          &amax_grad_out);
    DenseTensor x_sign;
    x_sign.Resize(in_x->dims());
    dev_ctx.template Alloc<T>(&x_sign);
    phi::SignKernel<T, Context>(dev_ctx, *in_x, &x_sign);
    phi::MultiplyKernel<T, Context>(dev_ctx, amax_grad_out, x_sign, out_dx);
  } else {
    auto functor = PNormGradFunctor<T>(porder, epsilon);
    funcs::LaunchReduceGradKernel<Context, T, PNormGradFunctor<T>>(
        dev_ctx, in_x, in_norm, in_norm_dy, out_dx, functor, dims, reduce_all);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(p_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::PNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
