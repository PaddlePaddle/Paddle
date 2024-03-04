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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename T, typename Context>
void BilinearKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    const DenseTensor& weight,
                    const paddle::optional<DenseTensor>& bias,
                    DenseTensor* out) {
  ctx.template Alloc<T>(out);

  auto y_mat = EigenMatrix<T>::From(y);
  auto output_mat = EigenMatrix<T>::From(*out);

  auto batch_size = x.dims()[0];
  auto weight_dims = weight.dims();
  int out_dim = weight_dims[0];
  auto x_dim = weight_dims[1];
  auto y_dim = weight_dims[2];
  auto& place = *ctx.eigen_device();

  // Create the intermediate variable to calculate the result of
  // Input(X) multiplied by Input(Weight_i), the formula is:
  // left_mul = X Weight_i.
  DenseTensor left_mul;
  left_mul.Resize(common::make_ddim({batch_size, y_dim}));
  ctx.template Alloc<T>(&left_mul);
  auto left_mul_mat = EigenMatrix<T>::From(left_mul);

  for (int i = 0; i < out_dim; ++i) {
    auto output_col_vec = output_mat.chip(i, 1);
    DenseTensor weight_mat =
        weight.Slice(i, i + 1).Resize(common::make_ddim({x_dim, y_dim}));
    phi::funcs::GetBlas<Context, T>(ctx).GEMM(CblasNoTrans,
                                              CblasNoTrans,
                                              batch_size,
                                              y_dim,
                                              x_dim,
                                              1,
                                              x.data<T>(),
                                              weight_mat.data<T>(),
                                              0,
                                              left_mul.data<T>());
    output_col_vec.device(place) =
        (left_mul_mat * y_mat).sum(Eigen::DSizes<int, 1>(1));
  }
  if (bias.get_ptr()) {
    auto bias_vec = EigenMatrix<T>::From(*(bias.get_ptr()));
    Eigen::DSizes<int, 2> bcast(batch_size, 1);
    output_mat.device(place) = bias_vec.broadcast(bcast) + output_mat;
  }
}

}  // namespace phi
