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

#include "paddle/phi/kernels/cholesky_solve_kernel.h"

#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
class CholeskySolveFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  bool upper,
                  int M,
                  int N,
                  T* Adata,
                  int lda,
                  T* Bdata,
                  int* devInfo);
};

template <typename T, typename Context>
void CholeskySolveKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         bool upper,
                         DenseTensor* out) {
  // get broadcast dim
  std::vector<int64_t> x_bst_dims_vec;
  std::vector<int64_t> y_bst_dims_vec;
  std::tie(x_bst_dims_vec, y_bst_dims_vec) =
      funcs::MatrixGetBroadcastDims(x, y);
  ScalarArray x_bst_dims(x_bst_dims_vec);
  ScalarArray y_bst_dims(y_bst_dims_vec);

  DenseTensor y_bst = phi::Empty<T, Context>(dev_ctx, y_bst_dims);
  ExpandKernel<T, Context>(dev_ctx, y, y_bst_dims, &y_bst);

  // Tensor broadcast to temp 'x_bst' and 'y_bst'
  DenseTensor x_bst = phi::Empty<T, Context>(dev_ctx, x_bst_dims);
  ExpandKernel<T, Context>(dev_ctx, x, x_bst_dims, &x_bst);

  // calculate y_bst's conjugate for complex
  DenseTensor y_bst_conj = Conj<T, Context>(dev_ctx, y_bst);
  y_bst_conj = phi::TransposeLast2Dim<T>(dev_ctx, y_bst_conj);
  T* y_bst_conj_data = y_bst_conj.data<T>();

  // calculate x_bst's conjugate for complex
  DenseTensor x_bst_conj = Conj<T, Context>(dev_ctx, x_bst);
  x_bst_conj = phi::TransposeLast2Dim<T>(dev_ctx, x_bst_conj);

  // copy x_bst's conjugate to 'result'
  DenseTensor result;
  Copy<Context>(dev_ctx, x_bst_conj, dev_ctx.GetPlace(), false, &result);
  T* res_data = result.data<T>();

  // CPU use lapack, GPU use cusolver
  int x_bst_ndim = x_bst_dims_vec.size();
  int M = static_cast<int>(x_bst_dims_vec[x_bst_ndim - 2]);
  int N = static_cast<int>(x_bst_dims_vec[x_bst_ndim - 1]);
  int batchsize = product(phi::slice_ddim(x_bst.dims(), 0, x_bst_ndim - 2));

  DenseTensor info =
      phi::Empty<int, Context>(dev_ctx, ScalarArray({batchsize}));
  int* info_data = info.data<int>();

  CholeskySolveFunctor<T, Context> functor;
  for (int i = 0; i < batchsize; ++i) {
    functor(dev_ctx,
            upper,
            M,
            N,
            y_bst_conj_data + i * M * M,
            std::max(1, M),
            res_data + i * M * N,
            info_data + i);
  }

  // calculate out's conjugate for complex
  result = phi::TransposeLast2Dim<T>(dev_ctx, result);
  out->Resize(phi::make_ddim(x_bst_dims_vec));
  ConjKernel<T, Context>(dev_ctx, result, out);
}

}  // namespace phi
