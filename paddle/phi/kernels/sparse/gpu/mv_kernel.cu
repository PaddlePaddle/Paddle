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

#include "paddle/phi/kernels/sparse/mv_kernel.h"

#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_blas.h"

namespace phi {
namespace sparse {

template <typename T, typename Context, typename TensorType>
void MvKernelImpl(const Context& dev_ctx,
                  const TensorType& x,
                  const DenseTensor& vec,
                  DenseTensor* out) {
#if CUDA_VERSION >= 11000
  std::vector<int64_t> x_dim = common::vectorize(x.dims());
  std::vector<int64_t> vec_dim = common::vectorize(vec.dims());
  auto x_ndims = x_dim.size();
  auto vec_ndims = vec_dim.size();
  PADDLE_ENFORCE_EQ(x_ndims,
                    2,
                    common::errors::InvalidArgument(
                        "the dims size of Input(x) must be equal to 2."));
  PADDLE_ENFORCE_EQ(vec_ndims,
                    1,
                    common::errors::InvalidArgument(
                        "the dims size of Input(vec) must be equal to 1."));
  PADDLE_ENFORCE_EQ(x_dim[x_ndims - 1],
                    vec_dim[vec_ndims - 1],
                    common::errors::PreconditionNotMet(
                        "The shape of Input(x) and Input(vec) is not "
                        "suitable for mv opetation, "
                        "x_dim[-1] must be equal to vec_dim[-1]."));
  std::vector<int64_t> out_dim = {x_dim[x_ndims - 2]};
  out->Resize(common::make_ddim(out_dim));
  dev_ctx.template Alloc<T>(out);
  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);
  sparse_blas.SPMV(false, static_cast<T>(1), x, vec, static_cast<T>(0), out);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      " 'sparse.mv' use cusparseSpMV, which is supported from CUDA 11.0"));
#endif
}

template <typename T, typename Context>
void MvCooKernel(const Context& dev_ctx,
                 const SparseCooTensor& x,
                 const DenseTensor& vec,
                 DenseTensor* out) {
  MvKernelImpl<T>(dev_ctx, x, vec, out);
}

template <typename T, typename Context>
void MvCsrKernel(const Context& dev_ctx,
                 const SparseCsrTensor& x,
                 const DenseTensor& vec,
                 DenseTensor* out) {
  MvKernelImpl<T>(dev_ctx, x, vec, out);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(
    mv_csr, GPU, ALL_LAYOUT, phi::sparse::MvCsrKernel, float, double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(
    mv_coo, GPU, ALL_LAYOUT, phi::sparse::MvCooKernel, float, double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
