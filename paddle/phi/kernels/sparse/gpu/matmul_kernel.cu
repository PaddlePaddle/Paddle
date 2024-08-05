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

#include "paddle/phi/kernels/sparse/matmul_kernel.h"

#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function_impl.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_blas.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_kernel_impl.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context, typename TensorType>
void MatmulKernelImpl(const Context& dev_ctx,
                      const TensorType& x,
                      const DenseTensor& y,
                      DenseTensor* out) {
#if CUDA_VERSION >= 11000 || HIP_VERSION >= 402
  std::vector<int64_t> xdim_vec = common::vectorize(x.dims());
  std::vector<int64_t> ydim_vec = common::vectorize(y.dims());
  auto x_ndims = xdim_vec.size();
  auto y_ndims = ydim_vec.size();
  PADDLE_ENFORCE_EQ(x_ndims,
                    y_ndims,
                    common::errors::PreconditionNotMet(
                        "The dims size of Input(x) and Input(y) "
                        "should be equal, But received X's "
                        "dimensions=%d, Y's dimensions=%d.",
                        x_ndims,
                        y_ndims));
  PADDLE_ENFORCE_GE(
      x_ndims,
      2,
      common::errors::InvalidArgument("the dims size of Input(x) and "
                                      "Input(y) must be greater than "
                                      "or equal to 2."));

  for (size_t i = 0; i < x_ndims - 2; ++i) {
    PADDLE_ENFORCE_EQ(xdim_vec[i],
                      ydim_vec[i],
                      common::errors::InvalidArgument(
                          "x.dim[%d] and x.dim[%d] must be eaqul.", i, i));
  }

  PADDLE_ENFORCE_GE(
      xdim_vec[x_ndims - 1],
      ydim_vec[y_ndims - 2],
      common::errors::PreconditionNotMet(
          "The shape of Input(x) and Input(y) is not suitable for matmul "
          "opetation, x_dim[-1] must be equal to y_dim[-2]."));

  // InferMeta of DenseTensor 'out'
  std::vector<int64_t> out_dim_vec(ydim_vec);
  out_dim_vec[y_ndims - 2] = xdim_vec[x_ndims - 2];
  out_dim_vec[y_ndims - 1] = ydim_vec[y_ndims - 1];
  MetaTensor meta_out(out);
  meta_out.set_dims(common::make_ddim(out_dim_vec));
  meta_out.set_dtype(y.dtype());

  dev_ctx.template Alloc<T>(out);

#ifdef PADDLE_WITH_HIP
  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, out, static_cast<T>(0.0f));
#endif

  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);
  sparse_blas.SPMM(
      false, false, static_cast<T>(1), x, y, static_cast<T>(0), out);
#else
#ifdef PADDLE_WITH_CUDA
  PADDLE_THROW(common::errors::Unimplemented(
      "forward of 'sparse.matmul' use cusparseSpMM, "
      "which is supported from CUDA 11.0"));
#elif defined(PADDLE_WITH_HIP)
  PADDLE_THROW(common::errors::Unimplemented(
      "forward of 'sparse.matmul' use rocsparse_spmm, "
      "which is supported from ROCM 4.2.0"));
#endif
#endif
}

template <typename T, typename Context>
void MatmulCooDenseKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  MatmulKernelImpl<T>(dev_ctx, x, y, out);
}

template <typename T, typename Context>
void MatmulCsrDenseKernel(const Context& dev_ctx,
                          const SparseCsrTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  MatmulKernelImpl<T>(dev_ctx, x, y, out);
}

template <typename T, typename Context>
void MatmulCsrCsrKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        const SparseCsrTensor& y,
                        SparseCsrTensor* out) {
#if CUDA_VERSION >= 11000
  std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
  std::vector<int64_t> ydim_vec = phi::vectorize(y.dims());
  auto x_ndims = xdim_vec.size();
  auto y_ndims = ydim_vec.size();
  PADDLE_ENFORCE_EQ(x_ndims,
                    y_ndims,
                    common::errors::PreconditionNotMet(
                        "The dims size of Input(x) and Input(y) "
                        "should be equal, But received X's "
                        "dimensions=%d, Y's dimensions=%d.",
                        x_ndims,
                        y_ndims));
  PADDLE_ENFORCE_GE(
      x_ndims,
      2,
      common::errors::InvalidArgument("the dims size of Input(x) and "
                                      "Input(y) must be greater than "
                                      "or equal to 2."));

  for (size_t i = 0; i < x_ndims - 2; ++i) {
    PADDLE_ENFORCE_EQ(xdim_vec[i],
                      ydim_vec[i],
                      common::errors::InvalidArgument(
                          "x.dim[%d] and x.dim[%d] must be eaqul.", i, i));
  }

  PADDLE_ENFORCE_GE(
      xdim_vec[x_ndims - 1],
      ydim_vec[y_ndims - 2],
      common::errors::PreconditionNotMet(
          "The shape of Input(x) and Input(y) is not suitable for matmul "
          "opetation, x_dim[-1] must be equal to y_dim[-2]."));

  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);
  sparse_blas.SPGEMM(
      false, false, static_cast<T>(1), x, y, static_cast<T>(0), out);

#else
#ifdef PADDLE_WITH_CUDA
  PADDLE_THROW(common::errors::Unimplemented(
      "forward of 'sparse.matmul' use cusparseSpGEMM, "
      "which is supported from CUDA 11.0"));
#endif
#endif
}

template <typename T, typename Context>
void MatmulCooCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const SparseCooTensor& y,
                        SparseCooTensor* out) {
  // 'cusparseSpGEMM' only support CSR now, so use COO->CSR->COO.
  SparseCsrTensor x_csr = CooToCsr<T, Context>(dev_ctx, x);
  SparseCsrTensor y_csr = CooToCsr<T, Context>(dev_ctx, y);
  SparseCsrTensor out_csr;
  out_csr.set_dims(out->dims());
  MatmulCsrCsrKernel<T>(dev_ctx, x_csr, y_csr, &out_csr);
  CsrToCooKernel<T>(dev_ctx, out_csr, out);
}

template <typename T, typename Context>
void MaskedMatmulCsrKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           const SparseCsrTensor& mask,
                           SparseCsrTensor* out) {
#if CUDA_VERSION >= 11030
  std::vector<int64_t> xdim_vec = common::vectorize(x.dims());
  std::vector<int64_t> ydim_vec = common::vectorize(y.dims());
  std::vector<int64_t> maskdim_vec = common::vectorize(mask.dims());

  auto x_ndims = xdim_vec.size();
  auto y_ndims = ydim_vec.size();
  auto mask_ndims = maskdim_vec.size();

  PADDLE_ENFORCE_EQ(x_ndims,
                    y_ndims,
                    common::errors::PreconditionNotMet(
                        "The dims size of Input(x) and Input(y) "
                        "should be equal, But received X's "
                        "dimensions=%d, Y's dimensions=%d.",
                        x_ndims,
                        y_ndims));
  PADDLE_ENFORCE_EQ(x_ndims,
                    mask_ndims,
                    common::errors::PreconditionNotMet(
                        "The dims size of Input(x) and Input(mask) "
                        "should be equal, But received X's "
                        "dimensions=%d, mask's dimensions=%d.",
                        x_ndims,
                        mask_ndims));
  PADDLE_ENFORCE_GE(
      x_ndims,
      2,
      common::errors::InvalidArgument("the dims size of Input(x) and "
                                      "Input(y) must be greater than "
                                      "or equal to 2."));

  for (size_t i = 0; i < x_ndims - 2; ++i) {
    PADDLE_ENFORCE_EQ(xdim_vec[i],
                      ydim_vec[i],
                      common::errors::InvalidArgument(
                          "x.dim[%d] and x.dim[%d] must match.", i, i));
    PADDLE_ENFORCE_EQ(xdim_vec[i],
                      maskdim_vec[i],
                      common::errors::InvalidArgument(
                          "x.dim[%d] and mask.dim[%d] must match.", i, i));
  }

  PADDLE_ENFORCE_GE(
      xdim_vec[x_ndims - 1],
      ydim_vec[y_ndims - 2],
      common::errors::PreconditionNotMet(
          "The shape of Input(x) and Input(y) is not suitable for matmul "
          "opetation, x_dim[-1] must be equal to y_dim[-2]."));

  PADDLE_ENFORCE_EQ(
      maskdim_vec[mask_ndims - 2],
      xdim_vec[x_ndims - 2],
      common::errors::PreconditionNotMet(
          "The shape of Input(x) and Input(y) is not suitable for matmul "
          "opetation, mask_dim[-2] must be equal to x_dim[-2]."));

  PADDLE_ENFORCE_EQ(
      maskdim_vec[mask_ndims - 1],
      ydim_vec[y_ndims - 1],
      common::errors::PreconditionNotMet(
          "The shape of Input(x) and Input(y) is not suitable for matmul "
          "opetation, mask_dim[-1] must be equal to y_dim[-1]."));

  // InferMeta of SparseCsrTensor 'out', CreateLikeInferMeta
  EmptyLikeCsrKernel<T, Context>(dev_ctx, mask, out);

  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);
  sparse_blas.SDDMM(
      false, false, static_cast<T>(1), x, y, static_cast<T>(0), out);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "forward of 'sparse.masked_matmul' use cusparseSDDMM, which is supported "
      "from CUDA 11.3"));
#endif
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(matmul_csr_dense,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MatmulCsrDenseKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(matmul_coo_dense,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MatmulCooDenseKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(matmul_coo_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MatmulCooCooKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(matmul_csr_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MatmulCsrCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(masked_matmul_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskedMatmulCsrKernel,
                   float,
                   double) {}
