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

#include "paddle/phi/kernels/triangular_solve_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void TriangularSolveKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           bool upper,
                           bool transpose,
                           bool unitriangular,
                           DenseTensor* out) {
  // get broadcast dim
  std::vector<int64_t> x_bst_dims_vec;
  std::vector<int64_t> y_bst_dims_vec;
  std::tie(x_bst_dims_vec, y_bst_dims_vec) =
      funcs::MatrixGetBroadcastDims(x, y);
  int x_bst_ndim = static_cast<int>(x_bst_dims_vec.size());
  int y_bst_ndim = static_cast<int>(y_bst_dims_vec.size());

  // Tensor broadcast to 'out' and temp 'x_bst'
  IntArray x_bst_dims(x_bst_dims_vec);
  DenseTensor x_bst = phi::Empty<T, Context>(dev_ctx, x_bst_dims);
  const T* x_bst_data = x_bst.data<T>();
  ExpandKernel<T, Context>(dev_ctx, x, x_bst_dims, &x_bst);

  out->Resize(common::make_ddim(y_bst_dims_vec));
  T* out_data = dev_ctx.template Alloc<T>(out);
  IntArray y_bst_dims(y_bst_dims_vec);
  ExpandKernel<T, Context>(dev_ctx, y, y_bst_dims, out);

  // Calculate use blas library
  int M = static_cast<int>(y_bst_dims_vec[y_bst_ndim - 2]);
  int N = static_cast<int>(y_bst_dims_vec[y_bst_ndim - 1]);
  int batch_size = 1;
  for (int i = 0; i < x_bst_ndim - 2; i++) {
    batch_size *= static_cast<int>(x_bst_dims_vec[i]);
  }

  auto blas = phi::funcs::GetBlas<CPUContext, T>(dev_ctx);
  for (int i = 0; i < batch_size; i++) {
    blas.TRSM(CblasLeft,
              upper ? CblasUpper : CblasLower,
              transpose ? CblasTrans : CblasNoTrans,
              unitriangular ? CblasUnit : CblasNonUnit,
              M,
              N,
              T(1),
              x_bst_data + i * M * M,
              std::max(1, M),
              out_data + i * N * M,
              std::max(1, N));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(triangular_solve,
                   CPU,
                   ALL_LAYOUT,
                   phi::TriangularSolveKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
