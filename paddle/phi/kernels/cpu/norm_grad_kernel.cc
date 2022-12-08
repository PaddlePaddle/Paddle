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

#include "paddle/phi/kernels/norm_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
namespace phi {

template <typename T, typename Context>
void NormGradKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& norm,
                    const DenseTensor& out_grad,
                    int axis,
                    float epsilon,
                    bool is_test,
                    DenseTensor* x_grad) {
  auto* in_x = &x;
  auto* in_dy = &out_grad;
  auto* in_norm = &norm;
  auto* out_dx = x_grad;

  ctx.template Alloc<T>(out_dx);

  auto xdim = in_x->dims();
  if (axis < 0) axis = xdim.size() + axis;
  int pre, n, post;
  funcs::GetPrePostNumel(xdim, axis, &pre, &n, &post);

  auto* place = ctx.eigen_device();

  auto x_e = phi::EigenVector<T>::Flatten(*in_x);
  auto dy_e = phi::EigenVector<T>::Flatten(*in_dy);
  auto norm_e = phi::EigenVector<T>::Flatten(*in_norm);
  auto dx_e = phi::EigenVector<T>::Flatten(*out_dx);

  Eigen::DSizes<int, 3> shape(pre, n, post);
  Eigen::DSizes<int, 3> rshape(pre, 1, post);
  auto x_r = x_e.reshape(shape);
  auto dy = dy_e.reshape(shape);
  auto norm_r = norm_e.reshape(rshape);
  auto dx = dx_e.reshape(shape);

  DenseTensor rsum;
  rsum.Resize({pre, post});
  ctx.template Alloc<T>(&rsum);
  auto sum = phi::EigenTensor<T, 2>::From(rsum);

  Eigen::DSizes<int, 1> rdim(1);
  Eigen::DSizes<int, 3> bcast(1, n, 1);

  // dx = ( dy/sqrt(sum(x*x)) ) * [1 - x*sum(x) / (sum(x*x) + e)]
  //    = [dy - dy * x * sum(x) / (sum(x*x) + e)] / sqrt(sum(x*x))
  //    = [dy - x * sum(x*dy) / (sum(x*x) + e)] / sqrt(sum(x*x))
  // 1. sum = sum(x*dy)
  sum.device(*place) = (x_r * dy).sum(rdim);
  // 2. dx = x * sum
  dx.device(*place) = sum.reshape(rshape).broadcast(bcast) * x_r;
  // 3. dx / (sum(x*x) + e)
  // where, norm.pow(2) = sum(x*x) + e, which is calculated in forward.
  dx.device(*place) = dx / norm_r.pow(2).broadcast(bcast);
  // 4. [dy - dx] / sqrt(sum(x*x))
  dx.device(*place) = (dy - dx) / norm_r.broadcast(bcast);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    norm_grad, CPU, ALL_LAYOUT, phi::NormGradKernel, float, double) {}
