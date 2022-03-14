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

#include "paddle/phi/kernels/addmm_kernel.h"

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
void AddmmKernel(const Context& dev_ctx,
                 const DenseTensor& input,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 float alpha,
                 float beta,
                 DenseTensor* out) {
  auto input_dims = input.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  // broadcast mode check
  if (x_dims[0] != input_dims[0]) {
    PADDLE_ENFORCE_EQ(input_dims[0],
                      1,
                      errors::InvalidArgument(
                          "When x_dims[0] is not equal with input_dims[0], "
                          "input_dims[0] must be 1 but got %s",
                          input_dims[0]));
    PADDLE_ENFORCE_EQ(y_dims[1] == input_dims[1] || input_dims[1] == 1,
                      true,
                      errors::InvalidArgument(
                          "The input tensor shape mismatch, input shape=[%s], "
                          "x shape=[%s], y shape=[%s]",
                          input_dims,
                          x_dims,
                          y_dims));
  }
  // broadcast mode check
  if (y_dims[1] != input_dims[1]) {
    PADDLE_ENFORCE_EQ(input_dims[1],
                      1,
                      errors::InvalidArgument(
                          "When y_dims[1] is not equal with input_dims[0], "
                          "input_dims[0] must be 1 but got %s",
                          input_dims[1]));
    PADDLE_ENFORCE_EQ(x_dims[0] == input_dims[0] || input_dims[0] == 1,
                      true,
                      errors::InvalidArgument(
                          "The input tensor shape mismatch, input shape=[%s], "
                          "x shape=[%s], y shape=[%s]",
                          input_dims,
                          x_dims,
                          y_dims));
  }
  // broadcast mode check
  PADDLE_ENFORCE_EQ(
      x_dims[1],
      y_dims[0],
      errors::InvalidArgument(
          "The input tensor X's width must be equal with matrix Y' height. "
          "But received X's shape = [%s], Y's shape = [%s].",
          x_dims[1],
          y_dims[0]));

  dev_ctx.template Alloc<T>(out);
  auto blas = funcs::GetBlas<Context, T>(dev_ctx);

  // calc broadcast dim
  Array2 bcast_dims;
  bcast_dims[0] = x_dims[0] / input_dims[0];
  bcast_dims[1] = y_dims[1] / input_dims[1];
  VLOG(3) << "bcast_dims=[" << bcast_dims[0] << "," << bcast_dims[1] << "]";
  // broadcast using eigen
  auto eigen_input = PhiEigenTensor<T, 2>::From(input);
  auto eigen_out = PhiEigenTensor<T, 2>::From(*out);
  auto& place = *dev_ctx.eigen_device();
  funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, 2>::Eval(
      place, eigen_out, eigen_input, bcast_dims);

  blas.GEMM(false,
            false,
            x_dims[0],
            y_dims[1],
            x_dims[1],
            alpha,
            x.data<T>(),
            x_dims[1],
            y.data<T>(),
            y_dims[1],
            beta,
            out->data<T>(),
            y_dims[1]);
}

}  // namespace phi
