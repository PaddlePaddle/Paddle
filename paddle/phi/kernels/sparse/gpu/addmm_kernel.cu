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

#include "paddle/phi/kernels/sparse/addmm_kernel.h"

#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_blas.h"

namespace phi {
namespace sparse {

template <typename T, typename Context, typename TensorType>
void AddmmKernelImpl(const Context& dev_ctx,
                     const DenseTensor& input,
                     const TensorType& x,
                     const DenseTensor& y,
                     float beta,
                     float alpha,
                     DenseTensor* out) {
#if CUDA_VERSION >= 11000
  std::vector<int64_t> input_dim = common::vectorize(input.dims());
  std::vector<int64_t> x_dim = common::vectorize(x.dims());
  std::vector<int64_t> y_dim = common::vectorize(y.dims());
  auto rank = input_dim.size();

  PADDLE_ENFORCE_GE(
      rank,
      2,
      common::errors::InvalidArgument(
          "the dims size of input must be greater than or equal to 2."));

  PADDLE_ENFORCE_EQ(
      x_dim.size(),
      rank,
      common::errors::PreconditionNotMet(
          "The dims size of Input(input) and Input(x) must be equal."));

  PADDLE_ENFORCE_GE(
      y_dim.size(),
      rank,
      common::errors::InvalidArgument(
          "the dims size of Input(input) and Input(y) must be equal."));

  for (size_t i = 0; i < rank - 2; ++i) {
    PADDLE_ENFORCE_EQ(input_dim[i],
                      x_dim[i],
                      common::errors::InvalidArgument(
                          "input.dim[%d] and x.dim[%d] must be eaqul.", i, i));
    PADDLE_ENFORCE_EQ(input_dim[i],
                      y_dim[i],
                      common::errors::InvalidArgument(
                          "input.dim[%d] and y.dim[%d] must be eaqul.", i, i));
  }

  PADDLE_ENFORCE_GE(
      input_dim[rank - 2],
      x_dim[rank - 2],
      common::errors::PreconditionNotMet(
          "The shape of Input(input) and Input(x) is not suitable for matmul "
          "opetation, input_dim[-2] must be equal to x_dim[-2]."));

  PADDLE_ENFORCE_GE(
      input_dim[rank - 1],
      y_dim[rank - 1],
      common::errors::PreconditionNotMet(
          "The shape of Input(input) and Input(y) is not suitable for matmul "
          "opetation, input_dim[-1] must be equal to y_dim[-1]."));

  PADDLE_ENFORCE_GE(
      x_dim[rank - 1],
      y_dim[rank - 2],
      common::errors::PreconditionNotMet(
          "The shape of Input(x) and Input(y) is not suitable for matmul "
          "opetation, x_dim[-1] must be equal to y_dim[-2]."));

  phi::Copy(dev_ctx, input, dev_ctx.GetPlace(), false, out);

  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);
  sparse_blas.SPMM(
      false, false, static_cast<T>(alpha), x, y, static_cast<T>(beta), out);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "forward of 'sparse.addmm' use cusparseSpMM, "
      "which is supported from CUDA 11.0"));
#endif
}

template <typename T, typename Context>
void AddmmCooDenseKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const SparseCooTensor& x,
                         const DenseTensor& y,
                         float beta,
                         float alpha,
                         DenseTensor* out) {
  AddmmKernelImpl<T>(dev_ctx, input, x, y, beta, alpha, out);
}

template <typename T, typename Context>
void AddmmCsrDenseKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const SparseCsrTensor& x,
                         const DenseTensor& y,
                         float beta,
                         float alpha,
                         DenseTensor* out) {
  AddmmKernelImpl<T>(dev_ctx, input, x, y, beta, alpha, out);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(addmm_coo_dense,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::AddmmCooDenseKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(addmm_csr_dense,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::AddmmCsrDenseKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
