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

namespace phi {

template <typename T, typename Context>
void BilinearGradKernel(const Context& ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& weight,
                        const DenseTensor& dout,
                        DenseTensor* dx,
                        DenseTensor* dy,
                        DenseTensor* dweight,
                        DenseTensor* dbias) {
  auto batch_size = x.dims()[0];
  auto weight_dims = weight.dims();
  int out_dim = weight_dims[0];
  auto x_dim = weight_dims[1];
  auto y_dim = weight_dims[2];

  auto x_mat = EigenMatrix<T>::From(x);
  auto y_mat = EigenMatrix<T>::From(y);
  auto dout_mat = EigenMatrix<T>::From(dout);
  auto& place = *ctx.eigen_device();
  // Create the intermediate variable to calculate the Output(Y@Grad).
  DenseTensor x_scale;
  x_scale.Resize(common::make_ddim({batch_size, x_dim}));
  ctx.template Alloc<T>(&x_scale);
  auto x_scale_mat = EigenMatrix<T>::From(x_scale);

  // Create the intermediate variable to calculate the Output(X@Grad).
  DenseTensor y_scale;
  y_scale.Resize(common::make_ddim({batch_size, y_dim}));
  ctx.template Alloc<T>(&y_scale);
  auto y_scale_mat = EigenMatrix<T>::From(y_scale);

  funcs::SetConstant<Context, T> set_zero;

  if (dx) {
    ctx.template Alloc<T>(dx);
    set_zero(ctx, dx, static_cast<T>(0));
  }

  if (dy) {
    ctx.template Alloc<T>(dy);
    set_zero(ctx, dy, static_cast<T>(0));
  }

  if (dweight) {
    ctx.template Alloc<T>(dweight);
  }

  auto blas = funcs::GetBlas<Context, T>(ctx);

  // Calculate the Output(X@Grad) and Output(Y@Grad).
  if (dx || dy || dweight) {
    Eigen::DSizes<int, 2> bcast_for_x(1, y_dim);
    Eigen::DSizes<int, 2> bcast_for_y(1, x_dim);
    Eigen::DSizes<int, 2> bcast_for_weight(1, x_dim);

    for (int i = 0; i < out_dim; ++i) {
      DenseTensor weight_i =
          weight.Slice(i, i + 1).Resize(common::make_ddim({x_dim, y_dim}));
      auto output_vec = dout_mat.chip(i, 1);

      if (dx) {
        y_scale_mat.device(place) =
            output_vec.reshape(Eigen::DSizes<int, 2>(batch_size, 1))
                .broadcast(bcast_for_x) *
            y_mat;
        blas.GEMM(CblasNoTrans,
                  CblasTrans,
                  batch_size,
                  x_dim,
                  y_dim,
                  1,
                  y_scale.data<T>(),
                  weight_i.data<T>(),
                  1,
                  dx->data<T>());
      }

      if (dy || dweight) {
        auto output_vec_y =
            output_vec.reshape(Eigen::DSizes<int, 2>(batch_size, 1))
                .broadcast(bcast_for_y);
        x_scale_mat.device(place) = output_vec_y * x_mat;
        if (dy) {
          blas.GEMM(CblasNoTrans,
                    CblasNoTrans,
                    batch_size,
                    y_dim,
                    x_dim,
                    1,
                    x_scale.data<T>(),
                    weight_i.data<T>(),
                    1,
                    dy->data<T>());
        }
        if (dweight) {
          DenseTensor dweight_i = dweight->Slice(i, i + 1).Resize(
              common::make_ddim({x_dim, y_dim}));
          blas.GEMM(CblasTrans,
                    CblasNoTrans,
                    x_dim,
                    y_dim,
                    batch_size,
                    1,
                    x_scale.data<T>(),
                    y.data<T>(),
                    0,
                    dweight_i.data<T>());
        }
      }
    }
  }

  // calculate the gradient of Input(Bias).
  if (dbias) {
    ctx.template Alloc<T>(dbias);
    auto dbias_mat = EigenVector<T>::Flatten(*dbias);
    dbias_mat.device(place) = dout_mat.sum(Eigen::DSizes<int, 1>(0));
  }
}

}  // namespace phi
