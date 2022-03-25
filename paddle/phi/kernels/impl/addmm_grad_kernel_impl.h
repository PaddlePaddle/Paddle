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

#include "paddle/phi/kernels/addmm_grad_kernel.h"

#include <type_traits>
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T,
          size_t D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using PhiEigenTensor = EigenTensor<T, D, MajorType, IndexType>;

using Array1 = Eigen::DSizes<Eigen::DenseIndex, 1>;
using Array2 = Eigen::DSizes<Eigen::DenseIndex, 2>;

template <typename T, typename Context>
void AddmmGradKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out_grad,
                     float alpha,
                     float beta,
                     DenseTensor* input_grad,
                     DenseTensor* x_grad,
                     DenseTensor* y_grad) {
  auto in_dims = input.dims();
  int total_elems = 0;

  VLOG(3) << "alpha: " << alpha << " beta: " << beta;

  if (input_grad != nullptr) {
    input_grad->set_lod(out_grad.lod());
  }
  if (x_grad != nullptr) {
    x_grad->set_lod(x.lod());
  }
  if (y_grad != nullptr) {
    y_grad->set_lod(y.lod());
  }

  auto blas = funcs::GetBlas<Context, T>(dev_ctx);
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    total_elems = in_dims[0] * in_dims[1];
    auto& place = *dev_ctx.eigen_device();
    auto eigen_dout = PhiEigenTensor<T, 2>::From(out_grad);
    auto eigen_dinput = PhiEigenTensor<T, 2>::From(*input_grad);

    bool row_compress = in_dims[0] != out_grad.dims()[0];
    bool col_compress = in_dims[1] != out_grad.dims()[1];
    auto eigen_dinput_shape =
        Array2(input_grad->dims()[0], input_grad->dims()[1]);

    if (row_compress && col_compress) {
      eigen_dinput.device(place) =
          eigen_dout.sum().eval().reshape(eigen_dinput_shape);
    } else if (row_compress) {
      eigen_dinput.device(place) =
          eigen_dout.sum(Array1(0)).eval().reshape(eigen_dinput_shape);
    } else if (col_compress) {
      eigen_dinput.device(place) =
          eigen_dout.sum(Array1(1)).eval().reshape(eigen_dinput_shape);
    } else {
      blas.VCOPY(total_elems, out_grad.data<T>(), input_grad->data<T>());
    }

    blas.SCAL(total_elems, beta, input_grad->data<T>());
  }
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
    total_elems = x.dims()[0] * x.dims()[1];
    // x_grad = out_grad * y'. x_grad: M x K, out_grad : M x N, y : K x N
    blas.MatMul(out_grad, false, y, true, x_grad);
    blas.SCAL(total_elems, alpha, x_grad->data<T>());
  }
  if (y_grad) {
    dev_ctx.template Alloc<T>(y_grad);
    total_elems = x.dims()[1] * y.dims()[1];
    // y_grad = x' * out_grad. y_grad K x N, out_grad : M x N, x : M x K
    blas.MatMul(x, true, out_grad, false, y_grad);
    blas.SCAL(total_elems, alpha, y_grad->data<T>());
  }
}

}  // namespace phi
