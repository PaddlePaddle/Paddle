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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

inline void GetDims(const phi::DDim& dim,
                    int axis,
                    int* pre,
                    int* n,
                    int* post,
                    bool asvector) {
  *pre = 1;
  *post = 1;
  *n = static_cast<int>(dim[axis]);
  if (asvector) {
    *n = static_cast<int>(product(dim));
  } else {
    for (int i = 0; i < axis; ++i) {
      (*pre) *= static_cast<int>(dim[i]);
    }
    for (int i = axis + 1; i < dim.size(); ++i) {
      (*post) *= static_cast<int>(dim[i]);
    }
  }
}

template <typename T, typename Context>
void PNormGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out,
                     const DenseTensor& out_grad,
                     float porder,
                     int axis,
                     float epsilon,
                     bool keepdim UNUSED,
                     bool asvector,
                     DenseTensor* x_grad) {
  auto* in_x = &x;
  auto* in_norm = &out;
  auto* in_norm_dy = &out_grad;
  auto* out_dx = x_grad;
  dev_ctx.template Alloc<T>(out_dx);

  T eps = static_cast<T>(epsilon);
  auto xdim = in_x->dims();

  if (axis < 0) axis = xdim.size() + axis;
  int pre, n, post;
  GetDims(xdim, axis, &pre, &n, &post, asvector);
  Eigen::DSizes<int, 3> shape(pre, n, post);
  Eigen::DSizes<int, 3> rshape(pre, 1, post);

  auto* place = dev_ctx.eigen_device();

  auto x_e = phi::EigenVector<T>::Flatten(*in_x);
  auto dx_e = phi::EigenVector<T>::Flatten(*out_dx);
  auto norm_e = phi::EigenVector<T>::Flatten(*in_norm);
  auto norm_dy_e = phi::EigenVector<T>::Flatten(*in_norm_dy);

  auto xr = x_e.reshape(shape);
  auto dx = dx_e.reshape(shape);
  auto norm = norm_e.reshape(rshape);
  auto norm_dy = norm_dy_e.reshape(rshape);

  Eigen::DSizes<int, 1> rdim(1);
  Eigen::DSizes<int, 3> bcast(1, n, 1);

  if (porder == 0) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, out_dx, static_cast<T>(0));
  } else if (porder == INFINITY || porder == -INFINITY) {
    dx.device(*place) = (xr.abs() == norm.broadcast(bcast)).template cast<T>() *
                        xr.sign() * norm_dy.broadcast(bcast);
  } else {
    dx.device(*place) =
        (xr.abs()).pow(porder - 1.0f) /
        ((norm.broadcast(bcast)).pow(porder - 1.0f) + xr.constant(eps));
    dx.device(*place) = dx * norm_dy.broadcast(bcast) * xr.sign();
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(
    p_norm_grad, CPU, ALL_LAYOUT, phi::PNormGradKernel, float, double) {}
