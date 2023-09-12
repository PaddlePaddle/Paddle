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
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/matrix_inverse.h"

namespace phi {

template <typename T>
struct IdentityMatrixFunctor {
  IdentityMatrixFunctor(const int m, T* output) : m_(m), output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int row = index / m_ % m_;
    const int col = index % m_;
    output_[index] = col == row ? static_cast<T>(1) : static_cast<T>(0);
  }

  const int m_;
  T* output_;
};

template <typename Context, typename T>
void MatrixPowerFunction(const DenseTensor* X,
                         const int n,
                         DenseTensor* Out,
                         const Context& ctx) {
  const auto& x_dims = X->dims();
  const int x_ndim = x_dims.size();
  T* out_data = ctx.template Alloc<T>(Out);

  phi::funcs::ForRange<Context> for_range(ctx, X->numel());

  if (n == 0) {
    // Out = Identity Matrix
    IdentityMatrixFunctor<T> functor(x_dims[x_ndim - 1], out_data);
    for_range(functor);
    return;
  }

  auto blas = phi::funcs::GetBlas<Context, T>(ctx);

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

  if (new_n == 1) {
    phi::Copy(ctx, new_x, ctx.GetPlace(), false, Out);
    return;
  }

  auto no_trans_desc = phi::funcs::CreateMatrixDescriptor(x_dims, 0, false);

  if (new_n == 2) {
    // Out = newX * newX
    ctx.template Alloc<T>(Out);
    blas.MatMul(new_x,
                no_trans_desc,
                new_x,
                no_trans_desc,
                static_cast<T>(1),
                Out,
                static_cast<T>(0));
    return;
  } else if (new_n == 3) {
    // Out = (newX * newX) * newX
    // Note: C[i] matrices in MatMul must not overlap, i.e. the individual
    // gemm operations must be computable independently; otherwise,
    // undefined behavior is expected.
    DenseTensor temp;
    temp.Resize(X->dims());
    ctx.template Alloc<T>(&temp);
    blas.MatMul(new_x,
                no_trans_desc,
                new_x,
                no_trans_desc,
                static_cast<T>(1),
                &temp,
                static_cast<T>(0));
    blas.MatMul(temp,
                no_trans_desc,
                new_x,
                no_trans_desc,
                static_cast<T>(1),
                Out,
                static_cast<T>(0));
    return;
  } else if (new_n == 4) {
    // Out = (newX * newX) * (newX * newX)
    DenseTensor temp;
    temp.Resize(X->dims());
    ctx.template Alloc<T>(&temp);
    blas.MatMul(new_x,
                no_trans_desc,
                new_x,
                no_trans_desc,
                static_cast<T>(1),
                &temp,
                static_cast<T>(0));
    blas.MatMul(temp,
                no_trans_desc,
                temp,
                no_trans_desc,
                static_cast<T>(1),
                Out,
                static_cast<T>(0));
    return;
  }

  // Calculate Out = newX^{n} for abs(n) > 4 with time complexity as O(logN)
  int bit = 0;
  DenseTensor z = DenseTensor(X->dtype());
  bool out_inited = false;
  DenseTensor temp_out;
  temp_out.Resize(X->dims());
  ctx.template Alloc<T>(&temp_out);
  DenseTensor temp_z;
  temp_z.Resize(X->dims());
  ctx.template Alloc<T>(&temp_z);
  while (new_n > 0) {
    bit = new_n & 0x1;
    new_n >>= 1;
    if (z.IsInitialized()) {
      blas.MatMul(z,
                  no_trans_desc,
                  z,
                  no_trans_desc,
                  static_cast<T>(1),
                  &temp_z,
                  static_cast<T>(0));
      phi::Copy(ctx, temp_z, ctx.GetPlace(), false, &z);
    } else {
      z.Resize(X->dims());
      ctx.template Alloc<T>(&z);
      phi::Copy(ctx, new_x, ctx.GetPlace(), false, &z);
    }
    if (bit == 1) {
      if (out_inited == true) {
        blas.MatMul(*Out,
                    no_trans_desc,
                    z,
                    no_trans_desc,
                    static_cast<T>(1),
                    &temp_out,
                    static_cast<T>(0));
        phi::Copy(ctx, temp_out, ctx.GetPlace(), false, Out);
      } else {
        phi::Copy(ctx, z, ctx.GetPlace(), false, Out);
        out_inited = true;
      }
    }
  }
  return;
}

template <typename T, typename Context>
void MatrixPowerKernel(const Context& ctx,
                       const DenseTensor& x,
                       int n,
                       DenseTensor* out) {
  const DenseTensor* X = &x;
  auto Out = out;

  const auto& x_dims = X->dims();
  const int x_ndim = x_dims.size();
  PADDLE_ENFORCE_EQ(
      x_dims[x_ndim - 2],
      x_dims[x_ndim - 1],
      errors::InvalidArgument(
          "The inner-most 2 dimensions of Input(X) should be equal."
          "X's shape[-2] = %d and shape[-1] = %d.",
          x_dims[x_ndim - 2],
          x_dims[x_ndim - 1]));

  MatrixPowerFunction<Context, T>(X, n, Out, ctx);
}

}  // namespace phi
