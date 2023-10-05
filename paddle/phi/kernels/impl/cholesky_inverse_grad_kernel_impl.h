// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/cholesky_inverse_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void CholeskyInverseGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& out,
                               const DenseTensor& out_grad,
                               bool upper,
                               DenseTensor* x_grad) {
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);

    auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

    DenseTensor tmp_out;
    tmp_out.Resize(out.dims());
    dev_ctx.template Alloc<T>(&tmp_out);

    auto mat_dim_a0 =
        phi::funcs::CreateMatrixDescriptor(out_grad.dims(), 0, false);
    auto mat_dim_b0 = phi::funcs::CreateMatrixDescriptor(out.dims(), 0, true);
    blas.MatMul(out_grad, mat_dim_a0, out, mat_dim_b0, T(1), &tmp_out, T(0));

    auto mat_dim_a1 = phi::funcs::CreateMatrixDescriptor(out.dims(), 0, true);
    auto mat_dim_b1 =
        phi::funcs::CreateMatrixDescriptor(tmp_out.dims(), 0, false);
    blas.MatMul(out, mat_dim_a1, tmp_out, mat_dim_b1, T(-1), x_grad, T(0));

    if (upper) {
      auto mat_dim_a2 = phi::funcs::CreateMatrixDescriptor(x.dims(), 0, false);
      auto mat_dim_b2 =
          phi::funcs::CreateMatrixDescriptor(x_grad->dims(), 0, false);
      blas.MatMul(x, mat_dim_a2, *x_grad, mat_dim_b2, T(1), &tmp_out, T(0));

      auto mat_dim_a3 = phi::funcs::CreateMatrixDescriptor(x.dims(), 0, false);
      auto mat_dim_b3 =
          phi::funcs::CreateMatrixDescriptor(x_grad->dims(), 0, true);
      blas.MatMul(x, mat_dim_a3, *x_grad, mat_dim_b3, T(1), x_grad, T(0));
      phi::AddKernel<T>(dev_ctx, tmp_out, *x_grad, x_grad);

    } else {
      auto mat_dim_a2 = phi::funcs::CreateMatrixDescriptor(x.dims(), 0, false);
      auto mat_dim_b2 =
          phi::funcs::CreateMatrixDescriptor(x_grad->dims(), 0, false);
      blas.MatMul(*x_grad, mat_dim_a2, x, mat_dim_b2, T(1), &tmp_out, T(0));

      auto mat_dim_a3 =
          phi::funcs::CreateMatrixDescriptor(x_grad->dims(), 0, true);
      auto mat_dim_b3 = phi::funcs::CreateMatrixDescriptor(x.dims(), 0, false);
      blas.MatMul(*x_grad, mat_dim_a3, x, mat_dim_b3, T(1), x_grad, T(0));
      phi::AddKernel<T>(dev_ctx, tmp_out, *x_grad, x_grad);
    }
  }
}
}  // namespace phi
