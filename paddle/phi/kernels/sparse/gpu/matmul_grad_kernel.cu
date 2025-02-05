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
#include "paddle/phi/kernels/funcs/math_function_impl.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_blas.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"
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
#if CUDA_VERSION >= 11030 || HIP_VERSION >= 403
  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);

  // dx{SparseCoo} = dout{Dense} * y'{Dense}
  if (dx) {
    // 'cusparseSDDMM' only support CSR now, so use COO->CSR->COO,
    // which will increase some expenses.
    EmptyLikeCooKernel<T, Context>(dev_ctx, x, dx);
    SparseCsrTensor dx_csr = CooToCsr<T, Context>(dev_ctx, *dx);
#ifdef PADDLE_WITH_HIP
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, dx_csr.mutable_non_zero_elements(), static_cast<T>(0.0f));
#endif
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

#ifdef PADDLE_WITH_HIP
    SparseCsrTensor x_csr = CooToCsr<T, Context>(dev_ctx, x);
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, dy, static_cast<T>(0.0f));
    sparse_blas.SPMM(
        true, false, static_cast<T>(1), x_csr, dout, static_cast<T>(0), dy);
#elif defined(PADDLE_WITH_CUDA)
    sparse_blas.SPMM(
        true, false, static_cast<T>(1), x, dout, static_cast<T>(0), dy);
#endif
  }
#else
#ifdef PADDLE_WITH_CUDA
  PADDLE_THROW(common::errors::Unimplemented(
      "backward of 'sparse.matmul' use cusparseSDDMM, which is supported from "
      "CUDA 11.3"));
#elif defined(PADDLE_WITH_HIP)
  PADDLE_THROW(
      common::errors::Unimplemented("backward of 'sparse.matmul' use "
                                    "rocsparse_sddmm with transpose, which is "
                                    "supported from "
                                    "ROCM 4.3.0"));
#endif
#endif
}

template <typename T, typename Context>
void MatmulCsrDenseGradKernel(const Context& dev_ctx,
                              const SparseCsrTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              SparseCsrTensor* dx,
                              DenseTensor* dy) {
#if CUDA_VERSION >= 11030 || HIP_VERSION >= 403
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

#ifdef PADDLE_WITH_HIP
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, dy, static_cast<T>(0.0f));
#endif

    sparse_blas.SPMM(
        true, false, static_cast<T>(1), x, dout, static_cast<T>(0), dy);
  }
#else
#ifdef PADDLE_WITH_CUDA
  PADDLE_THROW(common::errors::Unimplemented(
      "backward of 'sparse.matmul' use cusparseSDDMM, which is supported from "
      "CUDA 11.3"));
#elif defined(PADDLE_WITH_HIP)
  PADDLE_THROW(
      common::errors::Unimplemented("backward of 'sparse.matmul' use "
                                    "rocsparse_sddmm with transpose, which is "
                                    "supported from "
                                    "ROCM 4.3.0"));
#endif
#endif
}

template <typename T, typename Context>
void MatmulCsrCsrGradKernel(const Context& dev_ctx,
                            const SparseCsrTensor& x,
                            const SparseCsrTensor& y,
                            const SparseCsrTensor& dout,
                            SparseCsrTensor* dx,
                            SparseCsrTensor* dy) {
#if CUDA_VERSION >= 11000
  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);

  std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
  auto x_ndims = xdim_vec.size();
  std::vector<int> perm;
  if (x_ndims == 2) {
    perm = {1, 0};
  } else {
    perm = {0, 2, 1};
  }

  // dx{SparseCsr} = dout{SparseCsr} * y'{SparseCsr}
  if (dx) {
    // cusparseSpGEMM only support CUSPARSE_OPERATION_NON_TRANSPOSE.
    // transopse y before cusparseSpGEMM computation.
    SparseCsrTensor trans_y;
    TransposeCsrKernel<T, Context>(dev_ctx, y, perm, &trans_y);

    sparse_blas.SPGEMM(
        false, false, static_cast<T>(1), dout, trans_y, static_cast<T>(0), dx);
  }

  // dy{SparseCsr} = x'{SparseCsr} * dout{SparseCsr}
  if (dy) {
    // cusparseSpGEMM only support CUSPARSE_OPERATION_NON_TRANSPOSE.
    // transopse x before cusparseSpGEMM computation.
    SparseCsrTensor trans_x;
    TransposeCsrKernel<T, Context>(dev_ctx, x, perm, &trans_x);

    sparse_blas.SPGEMM(
        false, false, static_cast<T>(1), trans_x, dout, static_cast<T>(0), dy);
  }
#else
#ifdef PADDLE_WITH_CUDA
  PADDLE_THROW(common::errors::Unimplemented(
      "backward of 'sparse.matmul' use cusparseSpGEMM, which is supported from "
      "CUDA 11.0"));
#endif
#endif
}

template <typename T, typename Context>
void MatmulCooCooGradKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const SparseCooTensor& y,
                            const SparseCooTensor& dout,
                            SparseCooTensor* dx,
                            SparseCooTensor* dy) {
  // cusparseSpGEMM only support CSR now, so use COO->CSR->COO.
  SparseCsrTensor x_csr, y_csr, dout_csr, dx_csr, dy_csr;
  CooToCsrKernel<T>(dev_ctx, x, &x_csr);
  CooToCsrKernel<T>(dev_ctx, y, &y_csr);
  CooToCsrKernel<T>(dev_ctx, dout, &dout_csr);
  MetaTensor meta_dx_csr(&dx_csr);
  phi::UnchangedInferMeta(dx, &meta_dx_csr);
  MetaTensor meta_dy_csr(&dy_csr);
  phi::UnchangedInferMeta(dy, &meta_dy_csr);
  MatmulCsrCsrGradKernel<T>(dev_ctx, x_csr, y_csr, dout_csr, &dx_csr, &dy_csr);
  CsrToCooKernel<T>(dev_ctx, dx_csr, dx);
  CsrToCooKernel<T>(dev_ctx, dy_csr, dy);
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
    std::vector<int> trans_dim_vec = common::vectorize<int>(y.dims());
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

PD_REGISTER_KERNEL(matmul_csr_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MatmulCsrCsrGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(matmul_coo_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MatmulCooCooGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(masked_matmul_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskedMatmulCsrGradKernel,
                   float,
                   double) {}
