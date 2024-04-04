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

#include "paddle/phi/kernels/inverse_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/matrix_inverse.h"

namespace phi {

template <typename T, typename Context>
void InverseGradKernel(const Context& dev_ctx,
                       const DenseTensor& out,
                       const DenseTensor& out_grad,
                       DenseTensor* in_grad) {
  if (in_grad) {
    dev_ctx.template Alloc<T>(in_grad);

    auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

    DenseTensor tmp_out;
    tmp_out.Resize(out.dims());
    dev_ctx.template Alloc<T>(&tmp_out);

    if (IsComplexType(out.dtype())) {
      DenseTensor out_conj;
      out_conj.Resize(out.dims());
      dev_ctx.template Alloc<T>(&out_conj);

      phi::ConjKernel<T, Context>(dev_ctx, out, &out_conj);

      auto mat_dim_a0 =
          phi::funcs::CreateMatrixDescriptor(out_grad.dims(), 0, false);
      auto mat_dim_b0 = phi::funcs::CreateMatrixDescriptor(out.dims(), 0, true);
      blas.MatMul(
          out_grad, mat_dim_a0, out_conj, mat_dim_b0, T(1), &tmp_out, T(0));

      auto mat_dim_a1 = phi::funcs::CreateMatrixDescriptor(out.dims(), 0, true);
      auto mat_dim_b1 =
          phi::funcs::CreateMatrixDescriptor(tmp_out.dims(), 0, false);
      blas.MatMul(
          out_conj, mat_dim_a1, tmp_out, mat_dim_b1, T(-1), in_grad, T(0));
    } else {
      auto mat_dim_a0 =
          phi::funcs::CreateMatrixDescriptor(out_grad.dims(), 0, false);
      auto mat_dim_b0 = phi::funcs::CreateMatrixDescriptor(out.dims(), 0, true);
      blas.MatMul(out_grad, mat_dim_a0, out, mat_dim_b0, T(1), &tmp_out, T(0));

      auto mat_dim_a1 = phi::funcs::CreateMatrixDescriptor(out.dims(), 0, true);
      auto mat_dim_b1 =
          phi::funcs::CreateMatrixDescriptor(tmp_out.dims(), 0, false);
      blas.MatMul(out, mat_dim_a1, tmp_out, mat_dim_b1, T(-1), in_grad, T(0));
    }
  }
}

}  // namespace phi
