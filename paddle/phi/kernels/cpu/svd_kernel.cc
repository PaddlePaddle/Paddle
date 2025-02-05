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
#include "paddle/phi/kernels/svd_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T>
void LapackSvd(
    const T* X, T* U, T* VH, T* S, int rows, int cols, int full = false) {
  char jobz = full ? 'A' : 'S';
  int mx = std::max(rows, cols);
  int mn = std::min(rows, cols);
  T* a = const_cast<T*>(X);  // NOLINT
  int lda = rows;
  int ldu = rows;
  int ldvt = full ? cols : mn;
  int lwork = full ? (4 * mn * mn + 6 * mn + mx) : (4 * mn * mn + 7 * mn);
  std::vector<T> work(lwork);
  std::vector<int> iwork(8 * mn);
  int info = 0;
  phi::funcs::lapackSvd<T>(jobz,
                           rows,
                           cols,
                           a,
                           lda,
                           S,
                           U,
                           ldu,
                           VH,
                           ldvt,
                           work.data(),
                           lwork,
                           iwork.data(),
                           &info);
  if (info < 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "This %s-th argument has an illegal value", info));
  }
  if (info > 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "DBDSDC/SBDSDC did not converge, updating process failed. May be you "
        "passes a invalid matrix."));
  }
}

template <typename T>
void BatchSvd(const T* X,
              T* U,
              T* VH,
              T* S,
              int rows,
              int cols,
              int batches,
              int full = false) {
  // NOTE: this function is row major, because this function called the lapack.
  int stride = rows * cols;
  int k = std::min(rows, cols);
  int stride_u = full ? rows * rows : k * rows;
  int stride_v = full ? cols * cols : k * cols;
  for (int i = 0; i < batches; ++i) {
    LapackSvd<T>(X + i * stride,
                 U + i * stride_u,
                 VH + i * stride_v,
                 S + i * k,
                 rows,
                 cols,
                 full);
  }
  return;
}

template <typename T, typename Context>
void SvdKernel(const Context& dev_ctx,
               const DenseTensor& X,
               bool full_matrices,
               DenseTensor* U,
               DenseTensor* S,
               DenseTensor* VH) {
  int full = full_matrices;
  /*Create Tensors and output, set the dim ...*/
  auto numel = X.numel();
  DenseTensor trans_x = ::phi::TransposeLast2Dim<T>(dev_ctx, X);
  auto x_dims = X.dims();
  int rows = static_cast<int>(x_dims[x_dims.size() - 2]);
  int cols = static_cast<int>(x_dims[x_dims.size() - 1]);
  // int k = std::min(rows, cols);
  // int col_u = full ? rows : k;
  // int col_v = full ? cols : k;
  PADDLE_ENFORCE_LT(
      0,
      rows,
      errors::InvalidArgument("The row of Input(X) should be greater than 0."));
  PADDLE_ENFORCE_LT(
      0,
      cols,
      errors::InvalidArgument("The col of Input(X) should be greater than 0."));
  auto* x_data = trans_x.data<T>();
  int batches = static_cast<int>(numel / (rows * cols));
  auto* U_out = dev_ctx.template Alloc<phi::dtype::Real<T>>(U);
  auto* VH_out = dev_ctx.template Alloc<phi::dtype::Real<T>>(VH);
  auto* S_out = dev_ctx.template Alloc<phi::dtype::Real<T>>(S);
  /*SVD Use the Eigen Library*/
  BatchSvd<T>(x_data, U_out, VH_out, S_out, rows, cols, batches, full);
  /* let C[m, n] as a col major matrix with m rows and n cols.
   * let R[m, n] is row major matrix with m rows and n cols.
   * then we have: R[m,n] = C[m, n].resize((n,m)).transpose_last_two()
   * */
  auto col_major_to_row_major = [&dev_ctx](DenseTensor* out) {
    auto origin_dim = out->dims();
    int64_t& x = origin_dim[origin_dim.size() - 1];
    int64_t& y = origin_dim[origin_dim.size() - 2];
    std::swap(x, y);
    out->Resize(origin_dim);
    return ::phi::TransposeLast2Dim<T>(dev_ctx, *out);
  };
  *U = col_major_to_row_major(U);
  *VH = col_major_to_row_major(VH);
}

}  // namespace phi

PD_REGISTER_KERNEL(svd, CPU, ALL_LAYOUT, phi::SvdKernel, float, double) {}
