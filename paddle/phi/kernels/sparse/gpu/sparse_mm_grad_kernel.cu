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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_blas.h"
#include "paddle/phi/kernels/sparse/sparse_mm_grad_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void CsrDenseMatmulGradKernel(const Context& dev_ctx,
                              const SparseCsrTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              SparseCsrTensor* dx,
                              DenseTensor* dy) {
#if CUDA_VERSION >= 11030
  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);

  // dx{SparseCsr} = dout{Dense} * y'{Dense}
  if (dx) {
    // InferMeta of SparseCsrTensor 'dx'
    dx->set_dims(x.dims());

    phi::Copy(dev_ctx,
              x.non_zero_crows(),
              dev_ctx.GetPlace(),
              false,
              dx->mutable_non_zero_crows());
    phi::Copy(dev_ctx,
              x.non_zero_cols(),
              dev_ctx.GetPlace(),
              false,
              dx->mutable_non_zero_cols());

    DenseTensor* values = dx->mutable_non_zero_elements();
    values->Resize(x.non_zero_elements().dims());
    dev_ctx.template Alloc<T>(values);

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

    sparse_blas.DSDMM(
        true, false, static_cast<T>(1), x, dout, static_cast<T>(0), dy);
  }
#else
  PADDLE_THROW(phi::errors::Unimplemented(
      " backward of 'sparse.mm' use cusparseSDDMM, Only "
      "support it from CUDA 11.3"));
#endif
}

template <typename T, typename Context>
void CsrMaskedMatmulGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const SparseCsrTensor& dout,
                               DenseTensor* dx,
                               DenseTensor* dy) {
  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);

  // dx{Dense} = dout{SparseCsr} * y'{Dense}
  if (dx) {
    // InferMeta of DenseTensor 'dx'
    MetaTensor meta_dx(dx);
    meta_dx.set_dims(x.dims());
    meta_dx.set_dtype(x.dtype());

    dev_ctx.template Alloc<T>(dx);
    sparse_blas.DSDMM(
        false, true, static_cast<T>(1), dout, y, static_cast<T>(0), dx);
  }

  // dy{Dense} = x'{Dense} * dout{SparseCsr}
  // That is: dy'{Dense} = dout'{SparseCsr} * x{Dense}
  if (dy) {
    std::vector<int> trans_dim_vec = phi::vectorize<int>(y.dims());
    size_t rank = trans_dim_vec.size();
    std::swap(trans_dim_vec[rank - 1], trans_dim_vec[rank - 2]);
    DenseTensor trans_dy = phi::Empty<T, Context>(dev_ctx, trans_dim_vec);

    sparse_blas.DSDMM(
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
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(csr_dense_matmul_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrDenseMatmulGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(csr_masked_mm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrMaskedMatmulGradKernel,
                   float,
                   double) {}
