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

#include "paddle/phi/kernels/cholesky_solve_grad_kernel.h"

#include "paddle/phi/kernels/cholesky_solve_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/matrix_reduce.h"
#include "paddle/phi/kernels/funcs/tril_triu_compute.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void CholeskySolveGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& out,
                             const DenseTensor& dout,
                             bool upper,
                             DenseTensor* dx,
                             DenseTensor* dy) {
  // get broadcast dim
  std::vector<int64_t> x_bst_dims_vec;
  std::vector<int64_t> y_bst_dims_vec;
  std::tie(x_bst_dims_vec, y_bst_dims_vec) =
      funcs::MatrixGetBroadcastDims(x, y);
  IntArray x_bst_dims(x_bst_dims_vec);
  IntArray y_bst_dims(y_bst_dims_vec);

  // Tensor broadcast to temp 'y_bst'
  DenseTensor y_bst = phi::Empty<T, Context>(dev_ctx, y_bst_dims);
  ExpandKernel<T, Context>(dev_ctx, y, y_bst_dims, &y_bst);

  // reuse forward to calculate dx_bst, which is broad_cast of dx
  DenseTensor dx_bst = phi::Empty<T, Context>(dev_ctx, x_bst_dims);
  CholeskySolveKernel<T, Context>(dev_ctx, dout, y_bst, upper, &dx_bst);

  // get 'dx' according to 'dx_bst'
  dx->Resize(x.dims());
  dev_ctx.template Alloc<T>(dx);
  if (dx_bst.dims() == x.dims()) {
    Copy<Context>(dev_ctx, dx_bst, dev_ctx.GetPlace(), false, dx);
  } else {
    funcs::MatrixReduceSumFunctor<T, Context> functor;
    functor(dev_ctx, dx_bst, dx);
    dx->Resize(x.dims());
  }

  // calculate out's conjugate for complex
  DenseTensor out_conj = Conj<T, Context>(dev_ctx, out);
  out_conj = phi::TransposeLast2Dim<T>(dev_ctx, out_conj);

  DenseTensor commonterm = phi::Empty<T, Context>(dev_ctx, y_bst_dims);
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  blas.MatMul(dx_bst,
              phi::funcs::CreateMatrixDescriptor(dx_bst.dims(), 0, false),
              out_conj,
              phi::funcs::CreateMatrixDescriptor(out_conj.dims(), 0, false),
              static_cast<T>(1),
              &commonterm,
              static_cast<T>(0));

  // calculate commonterm's conjugate for complex
  DenseTensor commonterm_conj = Conj<T, Context>(dev_ctx, commonterm);
  commonterm_conj = phi::TransposeLast2Dim<T>(dev_ctx, commonterm_conj);

  phi::AddRawKernel<T>(dev_ctx, commonterm, commonterm_conj, -1, &commonterm);

  DenseTensor dy_bst = phi::Empty<T, Context>(dev_ctx, y_bst_dims);
  if (upper) {
    blas.MatMul(y_bst,
                phi::funcs::CreateMatrixDescriptor(y_bst.dims(), 0, false),
                commonterm,
                phi::funcs::CreateMatrixDescriptor(commonterm.dims(), 0, false),
                static_cast<T>(-1),
                &dy_bst,
                static_cast<T>(0));
  } else {
    blas.MatMul(commonterm,
                phi::funcs::CreateMatrixDescriptor(commonterm.dims(), 0, false),
                y_bst,
                phi::funcs::CreateMatrixDescriptor(y_bst.dims(), 0, false),
                static_cast<T>(-1),
                &dy_bst,
                static_cast<T>(0));
  }

  // get upper or lower of 'dy_bst'
  DenseTensor dy_bst_upper = phi::Empty<T, Context>(dev_ctx, y_bst_dims);

  int y_bst_ndim = y_bst_dims_vec.size();
  const auto H = y_bst_dims_vec[y_bst_ndim - 2];
  const auto W = y_bst_dims_vec[y_bst_ndim - 1];
  phi::funcs::ForRange<Context> y_for_range(dev_ctx, dy_bst.numel());
  phi::funcs::TrilTriuCompute<T> tril_triu_functor(
      dy_bst.data<T>(), 0, !upper, H, W, dy_bst_upper.data<T>());
  y_for_range(tril_triu_functor);

  // get 'dy' according to 'dy_bst'
  dy->Resize(y.dims());
  dev_ctx.template Alloc<T>(dy);
  if (dy_bst_upper.dims() == y.dims()) {
    Copy<Context>(dev_ctx, dy_bst_upper, dev_ctx.GetPlace(), false, dy);
  } else {
    funcs::MatrixReduceSumFunctor<T, Context> functor;
    functor(dev_ctx, dy_bst_upper, dy);
    dy->Resize(y.dims());
  }
}

}  // namespace phi
