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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/reduce_grad_functions.h"

namespace phi {

template <typename T>
struct AbsMaxAndMinGradFunctor {
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
    auto abs_x = (*x).abs();
    auto y_bc = y->broadcast(dim);
    auto dy_bc = dy->broadcast(dim);
    auto mask = (abs_x == y_bc).template cast<T>();

    Eigen::array<int, 1> reduce_dim = {static_cast<int>(dim.size() - 1)};
    auto shape1 = (*x).dimensions();
    shape1[shape1.size() - 1] = 1;
    auto shape2 = (*x).dimensions();
    for (size_t i = 0; i < shape2.size() - 1; i++) shape2[i] = 1;

    auto count = mask.sum(reduce_dim).reshape(shape1).broadcast(shape2);

    dx->device(place) = dy_bc * (*x).sign() * mask / count;
  }
};

template <typename T>
struct PNormGradFunctor {
  HOSTDEVICE explicit inline PNormGradFunctor(float porder, float eps) {
    this->porder = static_cast<T>(porder - 1.);
    this->eps = static_cast<T>(eps);
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
    dx->device(place) =
        (*x).abs().pow(this->porder) * (*x).sign() * dy->broadcast(dim) *
        (*y + y->constant(eps)).pow(-this->porder).broadcast(dim);
  }
  T porder;
  T eps;
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
  if (axis < 0) axis = xdim.size() + axis;
  const std::vector<int> dims = {axis};

  if (porder == 0) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, out_dx, static_cast<T>(0));
  } else if (porder == INFINITY || porder == -INFINITY) {
    AbsMaxAndMinGradFunctor<T> functor;
    funcs::LaunchReduceGradKernel<Context, T, AbsMaxAndMinGradFunctor<T>>(
        dev_ctx, in_x, in_norm, in_norm_dy, out_dx, functor, dims, reduce_all);

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
