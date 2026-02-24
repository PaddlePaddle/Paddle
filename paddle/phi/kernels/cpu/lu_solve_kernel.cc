// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/lu_solve_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/impl/lu_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void LuSolveKernel(const Context& dev_ctx,
                   const DenseTensor& b,
                   const DenseTensor& lu,
                   const DenseTensor& pivots,
                   const std::string& trans,
                   DenseTensor* out) {
  // Get lu matrix dimensions
  auto lu_dims = lu.dims();
  // Get x matrix dimensions
  auto x_dims = b.dims();

  // Allocate output tensor
  dev_ctx.template Alloc<T>(out);
  // Copy RHS data to output (will be overwritten with solution)
  *out = Transpose2DTo6D<Context, T>(dev_ctx, b);
  DenseTensor tem_lu = Transpose2DTo6D<Context, T>(dev_ctx, lu);

  // Prepare LAPACK parameters
  char trans_char = (trans == "N") ? 'N' : ((trans == "T") ? 'T' : 'C');
  auto n_last_dim = lu_dims[lu_dims.size() - 1];
  PADDLE_ENFORCE_LE_INT_MAX(
      n_last_dim,
      "TODO(large-tensor): LAPACK input n does not support int64 overflow.");
  int n_int = static_cast<int>(n_last_dim);

  auto nrhs_last_dim = x_dims[x_dims.size() - 1];
  PADDLE_ENFORCE_LE_INT_MAX(nrhs_last_dim,
                            "TODO(large-tensor): LAPACK nrhs does not "
                            "support int64 overflow.");
  int nrhs_int = static_cast<int>(nrhs_last_dim);
  int lda = std::max(1, n_int);  // Leading dimension of A (LU matrix)
  int ldb = std::max(1, n_int);  // Leading dimension of B (RHS/solution matrix)
  int info = 0;

  auto outdims = out->dims();
  auto outrank = outdims.size();
  auto batchsize_64 = product(slice_ddim(outdims, 0, outrank - 2));
  PADDLE_ENFORCE_LE_INT_MAX(
      batchsize_64,
      "TODO(large-tensor): LAPACK batch size does not support int64 overflow.");
  int batchsize = static_cast<int>(batchsize_64);
  auto out_data = out->data<T>();
  auto lu_data = tem_lu.data<T>();
  auto pivots_data =
      reinterpret_cast<int*>(const_cast<int*>(pivots.data<int>()));

  for (int i = 0; i < batchsize; i++) {
    auto* out_data_item = &out_data[i * lda * nrhs_int];
    auto* lu_data_item = &lu_data[i * ldb * n_int];
    auto* pivots_data_item = &pivots_data[i * n_int];
    funcs::lapackLuSolve<T>(trans_char,
                            n_int,
                            nrhs_int,
                            lu_data_item,
                            lda,
                            pivots_data_item,
                            out_data_item,
                            ldb,
                            &info);
    PADDLE_ENFORCE_EQ(
        info,
        0,
        common::errors::PreconditionNotMet(
            "LU solve failed with error code %d. Check if matrix is singular.",
            info));
  }
  *out = Transpose2DTo6D<Context, T>(dev_ctx, *out);
}
}  // namespace phi

PD_REGISTER_KERNEL(lu_solve,
                   CPU,
                   ALL_LAYOUT,
                   phi::LuSolveKernel,
                   float,
                   double,
                   phi::complex64,
                   phi::complex128) {}
