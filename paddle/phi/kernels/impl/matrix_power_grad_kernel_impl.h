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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/matrix_inverse.h"

namespace phi {

template <typename Context, typename T>
void MatrixPowerGradFunction(const DenseTensor* X,
                             const DenseTensor* Out,
                             const DenseTensor* dOut,
                             const int n,
                             DenseTensor* dX,
                             const Context& ctx) {
  ctx.template Alloc<T>(dX);
  const auto& x_dims = X->dims();

  auto blas = phi::funcs::GetBlas<Context, T>(ctx);

  if (n == 0) {
    // \nabla X = O
    phi::funcs::SetConstant<Context, T> zero;
    zero(ctx, dX, static_cast<T>(0));
    return;
  } else if (n == 1) {
    // \nabla X = \nabla Out
    phi::Copy(ctx, *dOut, ctx.GetPlace(), false, dX);
    return;
  }

  auto trans_desc = phi::funcs::CreateMatrixDescriptor(x_dims, 0, true);
  auto no_trans_desc = phi::funcs::CreateMatrixDescriptor(x_dims, 0, false);

  if (n == -1) {
    // \nabla X = Out^{T} * \nabla Out * Out^{T}
    DenseTensor temp_dx;
    temp_dx.Resize(X->dims());
    ctx.template Alloc<T>(&temp_dx);
    blas.MatMul(*Out,
                trans_desc,
                *dOut,
                no_trans_desc,
                static_cast<T>(-1),
                &temp_dx,
                static_cast<T>(0));
    blas.MatMul(temp_dx,
                no_trans_desc,
                *Out,
                trans_desc,
                static_cast<T>(1),
                dX,
                static_cast<T>(0));
    return;
  }

  DenseTensor new_x;
  new_x.Resize(X->dims());
  ctx.template Alloc<T>(&new_x);
  int new_n = n;
  if (n > 0) {
    // newX = X
    phi::Copy(ctx, *X, ctx.GetPlace(), false, &new_x);
  } else {
    // newX = X^{-1}, n = -n
    phi::funcs::MatrixInverseFunctor<Context, T> mat_inv;
    mat_inv(ctx, *X, &new_x);
    new_n = -n;
  }

  // Use chain rule blow to compute \nabla newX^{n}
  // First, Get newX^{0}, newX^{1}, ..., newX^{n - 1},
  // Note that newX^{0} can be omitted
  std::vector<std::shared_ptr<DenseTensor>> tensor_list(new_n - 1);
  tensor_list[0] = std::make_shared<DenseTensor>(new_x);
  int index = 1;
  while (index < new_n - 1) {
    DenseTensor tensor_list_index;
    tensor_list_index.Resize(X->dims());
    ctx.template Alloc<T>(&tensor_list_index);
    tensor_list[index] = std::make_shared<DenseTensor>(tensor_list_index);

    blas.MatMul(*tensor_list[index - 1],
                no_trans_desc,
                new_x,
                no_trans_desc,
                static_cast<T>(1),
                tensor_list[index].get(),
                static_cast<T>(0));
    index++;
  }

  // Second, \nabla newX = \sum_{i = 0}^{n - 1} (newX^{T}^{i}
  //                      * \nabla Out
  //                      * (newX^{T}^{n - i - 1})
  DenseTensor dx_new;
  dx_new.Resize(X->dims());
  ctx.template Alloc<T>(&dx_new);
  blas.MatMul(*tensor_list[new_n - 2],
              trans_desc,
              *dOut,
              no_trans_desc,
              static_cast<T>(1),
              &dx_new,
              static_cast<T>(0));
  DenseTensor da_an_minus1;
  da_an_minus1.Resize(X->dims());
  ctx.template Alloc<T>(&da_an_minus1);
  blas.MatMul(*dOut,
              no_trans_desc,
              *tensor_list[new_n - 2],
              trans_desc,
              static_cast<T>(1),
              &da_an_minus1,
              static_cast<T>(0));
  blas.AXPY(
      X->numel(), static_cast<T>(1), da_an_minus1.data<T>(), dx_new.data<T>());
  int start = 0;
  while (start < new_n - 2) {
    DenseTensor a_da;
    a_da.Resize(X->dims());
    ctx.template Alloc<T>(&a_da);
    DenseTensor a_da_a;
    a_da_a.Resize(X->dims());
    ctx.template Alloc<T>(&a_da_a);
    blas.MatMul(*tensor_list[start],
                trans_desc,
                *dOut,
                no_trans_desc,
                static_cast<T>(1),
                &a_da,
                static_cast<T>(0));
    blas.MatMul(a_da,
                no_trans_desc,
                *tensor_list[new_n - 3 - start],
                trans_desc,
                static_cast<T>(1),
                &a_da_a,
                static_cast<T>(0));
    blas.AXPY(
        X->numel(), static_cast<T>(1), a_da_a.data<T>(), dx_new.data<T>());
    start++;
  }

  if (n > 0) {
    // \nabla X = \nabla newX
    phi::Copy(ctx, dx_new, ctx.GetPlace(), false, dX);
  } else {
    // \nabla X = newX^{T} * \nabla newX * newX^{T}
    DenseTensor temp_dx;
    temp_dx.Resize(X->dims());
    ctx.template Alloc<T>(&temp_dx);
    blas.MatMul(new_x,
                trans_desc,
                dx_new,
                no_trans_desc,
                static_cast<T>(-1),
                &temp_dx,
                static_cast<T>(0));
    blas.MatMul(temp_dx,
                no_trans_desc,
                new_x,
                trans_desc,
                static_cast<T>(1),
                dX,
                static_cast<T>(0));
  }
  return;
}

template <typename T, typename Context>
void MatrixPowerGradKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& out,
                           const DenseTensor& out_grad,
                           int n,
                           DenseTensor* x_grad) {
  auto X = &x;
  auto Out = &out;
  auto dOut = &out_grad;
  auto dX = x_grad;

  MatrixPowerGradFunction<Context, T>(X, Out, dOut, n, dX, ctx);
}

}  // namespace phi
