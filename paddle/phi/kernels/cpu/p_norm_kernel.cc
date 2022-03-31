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

#include "paddle/phi/kernels/p_norm_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
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
  *n = dim[axis];
  if (asvector) {
    *n = product(dim);
  } else {
    for (int i = 0; i < axis; ++i) {
      (*pre) *= dim[i];
    }
    for (int i = axis + 1; i < dim.size(); ++i) {
      (*post) *= dim[i];
    }
  }
}

template <typename T, typename Context>
void PNormKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 float porder,
                 int axis,
                 float epsilon,
                 bool keepdim,
                 bool asvector,
                 DenseTensor* out) {
  auto* in_x = &x;
  dev_ctx.template Alloc<T>(out);

  auto xdim = in_x->dims();
  if (axis < 0) axis = xdim.size() + axis;
  int pre, n, post;
  GetDims(xdim, axis, &pre, &n, &post, asvector);

  auto* place = dev_ctx.eigen_device();

  Eigen::DSizes<int, 3> shape(pre, n, post);
  Eigen::DSizes<int, 2> norm_shape(pre, post);

  auto x_e = phi::EigenVector<T>::Flatten(*in_x);
  auto norm_e = phi::EigenVector<T>::Flatten(*out);

  auto xr = x_e.reshape(shape);
  auto norm = norm_e.reshape(norm_shape);

  // p=0 means number of non-zero elements of (xr)
  // p=inf means the maximum of |xr|
  // p=-inf means the minimum of |xr|
  // otherwise, Lp-norm = pow(sum(pow(|xr|, p)), 1/p)
  Eigen::DSizes<int, 1> rdim(1);
  if (porder == 0) {
    norm.device(*place) = (xr != xr.constant(0)).template cast<T>().sum(rdim);
  } else if (porder == INFINITY) {
    norm.device(*place) = xr.abs().maximum(rdim);
  } else if (porder == -INFINITY) {
    norm.device(*place) = xr.abs().minimum(rdim);
  } else {
    norm.device(*place) = xr.abs().pow(porder).sum(rdim).pow(1.0f / porder);
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(p_norm, CPU, ALL_LAYOUT, phi::PNormKernel, float, double) {}
