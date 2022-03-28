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

#include "paddle/phi/kernels/triangular_solve_grad_kernel.h"

#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/matrix_reduce.h"
#include "paddle/phi/kernels/funcs/tril_triu_compute.h"
#include "paddle/phi/kernels/triangular_solve_kernel.h"

namespace phi {

template <typename T, typename Context>
void TriangularSolveGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const DenseTensor& out,
                               const DenseTensor& dout,
                               bool upper,
                               bool transpose,
                               bool unitriangular,
                               DenseTensor* dx,
                               DenseTensor* dy) {
  std::vector<int64_t> x_bst_dims_vec;
  std::vector<int64_t> y_bst_dims_vec;
  std::tie(x_bst_dims_vec, y_bst_dims_vec) =
      funcs::MatrixGetBroadcastDims(x, y);

  ScalarArray y_bst_dims_array(y_bst_dims_vec);
  DenseTensor dy_bst = phi::Empty<T, Context>(dev_ctx, y_bst_dims_array);
  if (dy) {
    // calculate x's conjugate for complex
    DenseTensor x_conj;
    x_conj.Resize(x.dims());

    phi::funcs::ForRange<Context> x_for_range(dev_ctx, x.numel());
    phi::funcs::ConjFunctor<T> x_functor(
        x.data<T>(), x.numel(), dev_ctx.template Alloc<T>(&x_conj));
    x_for_range(x_functor);

    // reuse forward to get dy_bst, and the result has been broadcated already.
    TriangularSolveKernel<T, Context>(
        dev_ctx, x_conj, dout, upper, !transpose, unitriangular, &dy_bst);

    dy->Resize(y.dims());
    dev_ctx.template Alloc<T>(dy);
    if (dy_bst.dims() == y.dims()) {
      Copy<Context>(dev_ctx, dy_bst, dev_ctx.GetPlace(), false, dy);
    } else {
      funcs::MatrixReduceSumFunctor<T, Context> functor;
      functor(dev_ctx, dy_bst, dy);
      dy->Resize(y.dims());
    }
  }

  ScalarArray x_bst_dims_array(x_bst_dims_vec);
  DenseTensor dx_bst = phi::Empty<T, Context>(dev_ctx, x_bst_dims_array);
  if (dx) {
    // calculate x's conjugate for complex
    DenseTensor out_conj;
    out_conj.Resize(out.dims());

    phi::funcs::ForRange<Context> out_for_range(dev_ctx, out.numel());
    phi::funcs::ConjFunctor<T> out_functor(
        out.data<T>(), out.numel(), dev_ctx.template Alloc<T>(&out_conj));
    out_for_range(out_functor);

    auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
    if (transpose) {
      auto mat_dim_a =
          phi::funcs::CreateMatrixDescriptor(out_conj.dims(), 0, false);
      auto mat_dim_b =
          phi::funcs::CreateMatrixDescriptor(dy_bst.dims(), 0, true);
      blas.MatMul(out_conj,
                  mat_dim_a,
                  dy_bst,
                  mat_dim_b,
                  static_cast<T>(-1),
                  &dx_bst,
                  static_cast<T>(0));
    } else {
      auto mat_dim_a =
          phi::funcs::CreateMatrixDescriptor(dy_bst.dims(), 0, false);
      auto mat_dim_b =
          phi::funcs::CreateMatrixDescriptor(out_conj.dims(), 0, true);
      blas.MatMul(dy_bst,
                  mat_dim_a,
                  out_conj,
                  mat_dim_b,
                  static_cast<T>(-1),
                  &dx_bst,
                  static_cast<T>(0));
    }

    // get upper or lower triangular
    DenseTensor dx_bst_upper =
        phi::Empty<T, Context>(dev_ctx, x_bst_dims_array);

    const auto& dims = dx_bst.dims();
    const auto H = dims[dims.size() - 2];
    const auto W = dims[dims.size() - 1];
    phi::funcs::ForRange<Context> x_for_range(dev_ctx, dx_bst.numel());
    phi::funcs::TrilTriuCompute<T> tril_triu_functor(
        dx_bst.data<T>(), unitriangular, !upper, H, W, dx_bst_upper.data<T>());
    x_for_range(tril_triu_functor);

    dx->Resize(x.dims());
    dev_ctx.template Alloc<T>(dx);
    if (dx_bst.dims() == x.dims()) {
      Copy<Context>(dev_ctx, dx_bst_upper, dev_ctx.GetPlace(), false, dx);
    } else {
      funcs::MatrixReduceSumFunctor<T, Context> functor;
      functor(dev_ctx, dx_bst_upper, dx);
      dx->Resize(x.dims());
    }
  }
}

}  // namespace phi
