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

#include "paddle/phi/kernels/sparse/mv_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_blas.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT>
__global__ void MvCooGradGpuKernel(const T *dout,
                                   const T *vec,
                                   const IntT *dx_indices,
                                   T *dx_values,
                                   int nnz) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < nnz; idx += blockDim.x * gridDim.x) {
    int i = dx_indices[idx];
    int j = dx_indices[idx + nnz];
    dx_values[idx] = dout[i] * vec[j];
  }
}

template <typename T, typename IntT>
__global__ void MvCsrGradGpuKernel(const T *dout,
                                   const T *vec,
                                   const IntT *dx_crows,
                                   const IntT *dx_cols,
                                   T *dx_values,
                                   int row_number) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < row_number; i += gridDim.x * blockDim.x) {
    int row_first = static_cast<int>(dx_crows[i]);
    int row_nnz = static_cast<int>(dx_crows[i + 1] - dx_crows[i]);

    int non_zero_idx = blockIdx.y * blockDim.y + threadIdx.y;
    for (; non_zero_idx < row_nnz; non_zero_idx += gridDim.y * blockDim.y) {
      int j = dx_cols[row_first + non_zero_idx];
      dx_values[row_first + non_zero_idx] = dout[i] * vec[j];
    }
  }
}

template <typename T, typename Context>
void MvCooGradKernel(const Context &dev_ctx,
                     const SparseCooTensor &x,
                     const DenseTensor &vec,
                     const DenseTensor &dout,
                     SparseCooTensor *dx,
                     DenseTensor *dvec) {
  // dx{SparseCoo} = dout{Dense} * vec'{Dense}
  if (dx) {
    // InferMeta of SparseCooTensor 'dx', CreateLikeInferMeta
    EmptyLikeCooKernel<T, Context>(dev_ctx, x, dx);
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, dx->nnz());
    PD_VISIT_BASE_INTEGRAL_TYPES(
        dx->indices().dtype(), "MvCooGradKernel", ([&] {
          MvCooGradGpuKernel<T>
              <<<config.block_per_grid.x,
                 config.thread_per_block.x,
                 0,
                 dev_ctx.stream()>>>(dout.data<T>(),
                                     vec.data<T>(),
                                     dx->indices().data<data_t>(),
                                     dx->mutable_values()->data<T>(),
                                     dx->nnz());
        }));
  }

  // dvec{Dense} = x'{SparseCoo} * dout{Dense}
  if (dvec) {
#if CUDA_VERSION >= 11000
    // InferMeta of DenseTensor 'dvec'
    dvec->Resize(vec.dims());
    dev_ctx.template Alloc<T>(dvec);

    auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);
    sparse_blas.SPMV(true, static_cast<T>(1), x, dout, static_cast<T>(0), dvec);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        " vec.grad of 'sparse.mv' use cusparseSpMV, "
        "which is supported from CUDA 11.0"));
#endif
  }
}

template <typename T, typename Context>
void MvCsrGradKernel(const Context &dev_ctx,
                     const SparseCsrTensor &x,
                     const DenseTensor &vec,
                     const DenseTensor &dout,
                     SparseCsrTensor *dx,
                     DenseTensor *dvec) {
  // dx{SparseCsr} = dout{Dense} * vec'{Dense}
  if (dx) {
    // InferMeta of SparseCsrTensor 'dx', CreateLikeInferMeta
    EmptyLikeCsrKernel<T, Context>(dev_ctx, x, dx);

    int row_number = dx->dims()[0];
    int col_number = dx->dims()[1];
    auto config = phi::backends::gpu::GetGpuLaunchConfig2D(
        dev_ctx, col_number, row_number);
    PD_VISIT_BASE_INTEGRAL_TYPES(dx->crows().dtype(), "MvCsrGradKernel", ([&] {
                                   MvCsrGradGpuKernel<T>
                                       <<<config.block_per_grid.x,
                                          config.thread_per_block.x,
                                          0,
                                          dev_ctx.stream()>>>(
                                           dout.data<T>(),
                                           vec.data<T>(),
                                           dx->crows().data<data_t>(),
                                           dx->cols().data<data_t>(),
                                           dx->mutable_values()->data<T>(),
                                           row_number);
                                 }));
  }

  // dvec{Dense} = x'{SparseCsr} * dout{Dense}
  if (dvec) {
#if CUDA_VERSION >= 11000
    // InferMeta of DenseTensor 'dvec'
    dvec->Resize(vec.dims());
    dev_ctx.template Alloc<T>(dvec);

    auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);
    sparse_blas.SPMV(true, static_cast<T>(1), x, dout, static_cast<T>(0), dvec);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        " vec.grad of 'sparse.mv' use cusparseSpMV, "
        "which is supported from CUDA 11.0"));
#endif
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(
    mv_coo_grad, GPU, ALL_LAYOUT, phi::sparse::MvCooGradKernel, float, double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(
    mv_csr_grad, GPU, ALL_LAYOUT, phi::sparse::MvCsrGradKernel, float, double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
