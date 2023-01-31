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

#include "paddle/phi/kernels/sparse/matmul_grad_kernel.h"

#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_blas.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void MatmulCooDenseGradKernel(const Context& dev_ctx,
                              const SparseCooTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              SparseCooTensor* dx,
                              DenseTensor* dy) {
#if CUDA_VERSION >= 11030
  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);

  // dx{SparseCoo} = dout{Dense} * y'{Dense}
  if (dx) {
    // 'cusparseSDDMM' only support CSR now, so use COO->CSR->COO,
    // which will increase some expenses.
    EmptyLikeCooKernel<T, Context>(dev_ctx, x, dx);
    SparseCsrTensor dx_csr = CooToCsr<T, Context>(dev_ctx, *dx);
    sparse_blas.SDDMM(
        false, true, static_cast<T>(1), dout, y, static_cast<T>(0), &dx_csr);
    CsrToCooKernel<T, Context>(dev_ctx, dx_csr, dx);
  }

  // dy{Dense} = x'{SparseCoo} * dout{Dense}
  if (dy) {
    MetaTensor meta_dy(dy);
    meta_dy.set_dims(y.dims());
    meta_dy.set_dtype(y.dtype());
    dev_ctx.template Alloc<T>(dy);

    sparse_blas.SPMM(
        true, false, static_cast<T>(1), x, dout, static_cast<T>(0), dy);
  }
#else
  PADDLE_THROW(phi::errors::Unimplemented(
      "backward of 'sparse.matmul' use cusparseSDDMM, which is supported from "
      "CUDA 11.3"));
#endif
}

template <typename T, typename Context>
void MatmulCsrDenseGradKernel(const Context& dev_ctx,
                              const SparseCsrTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              SparseCsrTensor* dx,
                              DenseTensor* dy) {
#if CUDA_VERSION >= 11030
  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);

  // dx{SparseCsr} = dout{Dense} * y'{Dense}
  if (dx) {
    // InferMeta of SparseCsrTensor 'dx', CreateLikeInferMeta
    EmptyLikeCsrKernel<T, Context>(dev_ctx, x, dx);

    sparse_blas.SDDMM(
        false, true, static_cast<T>(1), dout, y, static_cast<T>(0), dx);
  }

  // dy{Dense} = x'{SparseCsr} * dout{Dense}
  if (dy) {
    // InferMeta of DenseTensor 'dy'
    MetaTensor meta_dy(dy);
    meta_dy.set_dims(y.dims());
    meta_dy.set_dtype(y.dtype());

    dev_ctx.template Alloc<T>(dy);

    sparse_blas.SPMM(
        true, false, static_cast<T>(1), x, dout, static_cast<T>(0), dy);
  }
#else
  PADDLE_THROW(phi::errors::Unimplemented(
      "backward of 'sparse.matmul' use cusparseSDDMM, which is supported from "
      "CUDA 11.3"));
#endif
}

template <typename T, typename Context>
void MaskedMatmulCsrGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const SparseCsrTensor& dout,
                               DenseTensor* dx,
                               DenseTensor* dy) {
#if CUDA_VERSION >= 11000
  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);

  // dx{Dense} = dout{SparseCsr} * y'{Dense}
  if (dx) {
    // InferMeta of DenseTensor 'dx'
    MetaTensor meta_dx(dx);
    meta_dx.set_dims(x.dims());
    meta_dx.set_dtype(x.dtype());

    dev_ctx.template Alloc<T>(dx);
    sparse_blas.SPMM(
        false, true, static_cast<T>(1), dout, y, static_cast<T>(0), dx);
  }

  // dy{Dense} = x'{Dense} * dout{SparseCsr}
  // That is: dy'{Dense} = dout'{SparseCsr} * x{Dense}
  if (dy) {
    std::vector<int> trans_dim_vec = phi::vectorize<int>(y.dims());
    size_t rank = trans_dim_vec.size();
    std::swap(trans_dim_vec[rank - 1], trans_dim_vec[rank - 2]);
    DenseTensor trans_dy = phi::Empty<T, Context>(dev_ctx, trans_dim_vec);

    sparse_blas.SPMM(
        true, false, static_cast<T>(1), dout, x, static_cast<T>(0), &trans_dy);

    // InferMeta of DenseTensor 'dy'
    MetaTensor meta_dy(dy);
    meta_dy.set_dims(y.dims());
    meta_dy.set_dtype(y.dtype());

    dev_ctx.template Alloc<T>(dy);

    size_t y_ndim = y.dims().size();
    std::vector<int> axis(y_ndim);
    for (size_t i = 0; i < y_ndim; ++i) {
      axis[i] = i;
    }
    std::swap(axis[y_ndim - 1], axis[y_ndim - 2]);
    TransposeKernel<T, Context>(dev_ctx, trans_dy, axis, dy);
  }
#endif
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(matmul_coo_dense_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MatmulCooDenseGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(matmul_csr_dense_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MatmulCsrDenseGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(masked_matmul_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskedMatmulCsrGradKernel,
                   float,
                   double) {}
